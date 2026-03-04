#include "common/detector.hh"

#include "common/bd_reader.hh"
#include "common/detector_utils.hh"

#include "detector.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TransitionId.hh"

#include "httplib.h"
#include "rapidjson/document.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
//#include <mdspan>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdfloat>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace XTCPP {
  namespace Base {
    Detector::Detector(std::string detname,
                       std::string serial_no,
                       std::vector<unsigned> segment_nos,
                       std::vector<std::shared_ptr<BDReader>> xtc_readers,
                       std::string experiment,
                       std::string run,
                       bool is_epics,
                       bool is_scan)
      : m_detname(detname)
      , m_serial_no(serial_no)
      , m_det_type(xtc_readers[0]->det_types()[m_detname])
      , m_experiment(experiment)
      , m_run(run)
      , m_segment_nos(segment_nos)
      , m_xtc_readers(xtc_readers)
      , m_data_ptrs(m_segment_nos.size())
      , m_data_sizes(m_segment_nos.size())
      , m_is_epics(is_epics)
      , m_is_scan(is_scan)
    {
      if (auto tmp = spdlog::get("Base::Detector")) {
        m_logger = tmp;
      } else {
        m_logger = spdlog::stdout_color_mt("Base::Detector");
      }
      if (!m_is_epics && !m_is_scan) {
        // Only needed for detectors with calibration constants
        get_detector_short_name();
      }

      const char* det_read_mode = std::getenv("XTCPP_DET_GETDATA");
      if (det_read_mode && std::string(det_read_mode) == "THREADED") {
        get_l1_data_impl = &Detector::get_l1_data_threaded;
        get_l1_data_op_impl = &Detector::get_l1_data_threaded_op;
        // ThreadPool is not copyable/moveable - construct in place
        m_thread_pool.emplace(m_xtc_readers.size());
        m_logger->debug("Will use threaded file read mode for get_data.");
      } else {
        get_l1_data_impl = &Detector::get_l1_data_sequential;
        get_l1_data_op_impl = &Detector::get_l1_data_sequential_op;
        m_thread_pool = std::nullopt;
        m_logger->debug("Will use sequential file read mode for get_data.");
      }

      auto& reader = m_xtc_readers[0];
      if (m_is_epics) {

        m_det_algs = reader->det_algs()["epics"];
        m_det_alg_fields = reader->det_alg_fields()["epics"];
      } else {
        m_det_algs = reader->det_algs()[m_detname];
        m_det_alg_fields = reader->det_alg_fields()[m_detname];
      }

      if (m_det_type == "jungfrau") {
        m_calib_data = std::vector<std::float32_t>(32 * 512 * 1024);
      } else if (m_det_type == "epix100") {
        m_calib_data = std::vector<std::float32_t>(704 * 768);
      }
    }

    void Detector::load_all_calib_constants() {
      // Database all refers to shortname, without cannot get constants
      m_logger->info("*** Starting to load calibration constants ***");
      auto [peds, shape_peds] = load_calib_constants_type("pedestals");
      auto [offs, shape_offs] = load_calib_constants_type("pixel_offset");
      auto [gain, shape_gain] = load_calib_constants_type("pixel_gain");

      // Assume for now they are all the same shape
      size_t total_entries = std::reduce(shape_peds.begin(),
                                         shape_peds.end(),
                                         1,
                                         std::multiplies{});
      m_logger->trace("Constructing m_calibconst vector with {} entries",
                      total_entries);
      m_calibconst.resize(total_entries);
      for (size_t idx=0; idx < total_entries; ++idx) {
        auto ped_off = peds[idx];
        if (!offs.empty()) {
          ped_off += offs[idx];
        }
        std::float32_t gain_val;
        if (!gain.empty()) {
          gain_val = gain[idx];
        } else {
          gain_val = 1.0f;
        }
        m_calibconst[idx] = CalibStruct(ped_off, gain_val);
      }
      m_logger->info("*** Done loading calibration constants ***");
    }

    void Detector::get_detector_short_name() {
      httplib::Client cli("https://pswww.slac.stanford.edu");

      std::string detnames_endpoint = "/calib_ws/cdb_detnames/" + m_det_type;

      m_logger->trace("Searching for `shortname` for " + m_detname);
      if (auto res = cli.Get(detnames_endpoint)) {
        rapidjson::Document docs;
        docs.Parse(res->body.c_str());
        if (!docs.IsArray()) {
          std::cerr << " Not a list of docs! " << std::endl;
          return;
        }

        for (const auto& doc : docs.GetArray()) {

          if (!doc.IsObject()) continue;

          std::string ser_no = doc["long"].GetString();
          if (ser_no == m_serial_no) {
            m_short_name = doc["short"].GetString();
            m_logger->debug("Found short name `" + m_short_name + "` for serial number: "
                            + m_serial_no);
            return;
          }
        }
      } else {
        m_logger->error("Could not find shortname for " + m_detname);
        return;
      }
    }

    std::vector<std::string> Detector::split_string(const std::string& s,
                                                    const std::string& delim) {
      std::vector<std::string> parts;
      size_t nextPos{0};
      size_t lastPos{0};

      std::string part;
      while ((nextPos = s.find(delim, lastPos)) != std::string::npos) {
        part = s.substr(lastPos,nextPos - lastPos);
        if (!part.empty()) {
          parts.push_back(part);
        }
        lastPos = nextPos + 1;
      }
      part = s.substr(lastPos);
      parts.push_back(part);
      return parts;
    }

    std::pair<std::vector<std::float32_t>,std::vector<size_t>>
    Detector::load_calib_constants_type(std::string constants_type) {

      std::string data_type{""}; // Whether an array etc
      std::string data_dtype{""}; // If an array whats the dtype (e.g. float32)
      size_t data_ndim{0}; // Number of dimensions
      size_t data_size{0}; // Total number of pixels
      std::vector<size_t> data_shape(0); // Shape
      std::vector<std::float32_t> constants;
      if (m_short_name.empty()) {
        m_logger->error("Cannot load constants without short name!");
        return std::make_pair(constants,data_shape);
      }
      httplib::Client cli("https://pswww.slac.stanford.edu");

      std::string db_in_use = "cdb_" + m_experiment;

      std::string data_doc_id{""};
      std::string exp_endpoint = "/calib_ws/" + db_in_use + "/" + m_short_name;
      m_logger->debug("Getting " + constants_type + " for " + m_short_name);
      m_logger->debug("Trying experiment endpoint: " + exp_endpoint);
      if (auto res = cli.Get(exp_endpoint)) {
        rapidjson::Document docs;
        docs.Parse(res->body.c_str());
        if (!docs.IsArray()) {
          m_logger->error("Not a list of docs from " + exp_endpoint);
        } else {
          for (const auto& doc : docs.GetArray()) {
            if (!doc.IsObject()) {
              continue;
            }
            std::string doc_constants_type = doc["ctype"].GetString();
            if (constants_type != doc_constants_type) {
              continue;
            }

            // Really --- Here would need to do the run validity checks!!!
            data_doc_id = doc["id_data"].GetString();
            m_logger->debug("Selected data ID: " + data_doc_id);
            data_type = doc["data_type"].GetString();
            data_dtype = doc["data_dtype"].GetString();
            data_ndim = static_cast<size_t>(std::atoi(doc["data_ndim"].GetString()));
            data_size = static_cast<size_t>(std::atoi(doc["data_size"].GetString()));

            std::string data_shape_str = doc["data_shape"].GetString();
            data_shape_str = data_shape_str.substr(1, data_shape_str.size()-2);
            std::vector<std::string> shape_parts = split_string(data_shape_str, ",");
            data_shape.resize(data_ndim);
            for (size_t i=0; i < data_ndim; i++) {
              data_shape[i] = static_cast<size_t>(std::stoi(shape_parts[i]));
            }
            break;
          }
        }
      } else {
        db_in_use = "cdb_" + m_short_name;
        std::string det_endpoint = "/calib_ws/" + db_in_use + "/" + m_short_name;
        m_logger->debug("Trying det endpoint (exp doesn't exist): " + det_endpoint);
        if (auto res = cli.Get(det_endpoint)) {
          rapidjson::Document docs;
          docs.Parse(res->body.c_str());
          if (!docs.IsArray()) {
            m_logger->error("Not a list of docs from " + exp_endpoint);
          } else {
            for (const auto& doc : docs.GetArray()) {
              if (!doc.IsObject()) {
                continue;
              }
              std::string doc_constants_type = doc["ctype"].GetString();
              if (constants_type != doc_constants_type) {
                continue;
              }

              // Really --- Here would need to do the run validity checks!!!
              data_doc_id = doc["id_data"].GetString();
              m_logger->debug("Selected data ID: " + data_doc_id);
              data_type = doc["data_type"].GetString();
              data_dtype = doc["data_dtype"].GetString();
              data_ndim = static_cast<size_t>(std::atoi(doc["data_ndim"].GetString()));
              data_size = static_cast<size_t>(std::atoi(doc["data_size"].GetString()));

              std::string data_shape_str = doc["data_shape"].GetString();
              data_shape_str = data_shape_str.substr(1, data_shape_str.size()-2);
              std::vector<std::string> shape_parts = split_string(data_shape_str, ",");
              data_shape.resize(data_ndim);
              for (size_t i=0; i < data_ndim; i++) {
                data_shape[i] = static_cast<size_t>(std::stoi(shape_parts[i]));
              }
              break;
            }
          }
        }
      }
      if (data_doc_id.empty()) {
        db_in_use = "cdb_" + m_short_name;
        std::string det_endpoint = "/calib_ws/" + db_in_use + "/" + m_short_name;
        m_logger->debug("Trying det endpoint (data id not found): " + det_endpoint);
        if (auto res = cli.Get(det_endpoint)) {
          rapidjson::Document docs;
          docs.Parse(res->body.c_str());
          if (!docs.IsArray()) {
            m_logger->error("Not a list of docs from " + exp_endpoint);
          } else {
            for (const auto& doc : docs.GetArray()) {
              if (!doc.IsObject()) {
                continue;
              }
              std::string doc_constants_type = doc["ctype"].GetString();
              if (constants_type != doc_constants_type) {
                continue;
              }

              // Really --- Here would need to do the run validity checks!!!
              data_doc_id = doc["id_data"].GetString();
              m_logger->debug("Selected data ID: " + data_doc_id);
              data_type = doc["data_type"].GetString();
              data_dtype = doc["data_dtype"].GetString();
              data_ndim = static_cast<size_t>(std::atoi(doc["data_ndim"].GetString()));
              data_size = static_cast<size_t>(std::atoi(doc["data_size"].GetString()));

              std::string data_shape_str = doc["data_shape"].GetString();
              data_shape_str = data_shape_str.substr(1, data_shape_str.size()-2);
              std::vector<std::string> shape_parts = split_string(data_shape_str, ",");
              data_shape.resize(data_ndim);
              for (size_t i=0; i < data_ndim; i++) {
                data_shape[i] = static_cast<size_t>(std::stoi(shape_parts[i]));
              }
              break;
            }
          }
        }
      }
      m_logger->trace("Selected data has {} dims and total size = {}",
                      data_ndim,
                      data_size);
      std::string shape_msg{"("};
      for (size_t i=0; i<data_shape.size(); ++i) {
        shape_msg += std::to_string(data_shape[i]);
        if (i < data_shape.size() - 1) {
          shape_msg += ", ";
        }
      }
      shape_msg += ")";
      m_logger->trace("Data shape is: {}", shape_msg);
      // Now get the data using data_doc_id
      std::string data_endpoint = "/calib_ws/" + db_in_use + "/gridfs/" + data_doc_id;
      if (auto res = cli.Get(data_endpoint)) {
        if (data_type == "ndarray") {
          auto* raw_data = reinterpret_cast<const unsigned char*>(res->body.data());
          constants.resize(data_size);
          if (data_dtype == "float32") {
            m_logger->debug("Getting float32 ndarray");
            std::memcpy(constants.data(), raw_data, data_size*sizeof(std::float32_t));
          } else if (data_dtype == "float64") {
            m_logger->debug("Getting float64 ndarray");
            std::vector<std::float64_t> temp(data_size);
            std::memcpy(temp.data(), raw_data, data_size*sizeof(std::float64_t));
            std::transform(temp.begin(),temp.end(), constants.begin(),
                           [](std::float64_t val) {
                             return static_cast<std::float32_t>(val);
                           });
          }
        }
      }
      return std::make_pair(constants, data_shape);
    }

    void Detector::load_dummy_calib() {
      if (m_detname == "jungfrau") {
        std::ifstream peds_in("peds.npy", std::ios::binary);
        std::ifstream gain_in("gain.npy", std::ios::binary);
        std::ifstream offs_in("offset.npy", std::ios::binary);

        std::vector<std::float32_t> peds(3*32*512*1024);
        std::vector<std::float32_t> gain(3*32*512*1024);
        std::vector<std::float32_t> offs(3*32*512*1024);

        peds_in.read(reinterpret_cast<char*>(peds.data()),peds.size()*sizeof(std::float32_t));
        gain_in.read(reinterpret_cast<char*>(gain.data()),gain.size()*sizeof(std::float32_t));
        offs_in.read(reinterpret_cast<char*>(offs.data()),offs.size()*sizeof(std::float32_t));

        m_calibconst.resize(3*32*512*1024);
        for (size_t gain_idx=0; gain_idx < 3; ++gain_idx) {
          for (size_t seg_idx=0; seg_idx < 32; ++seg_idx) {
            for (size_t row_idx=0; row_idx < 512; ++row_idx) {
              for (size_t col_idx=0; col_idx < 1024; ++col_idx) {
                size_t idx =
                  gain_idx*32*512*1024 +
                  seg_idx*512*1024 +
                  row_idx*1024 +
                  col_idx;
                auto ped_off = peds[idx] + offs[idx];
                m_calibconst[idx] = CalibStruct(ped_off, gain[idx]);
              }
            }
          }
        }
      }
    }

    const XtcData::Dgram* const Detector::operator()(size_t offset_idx) {
      auto& reader = m_xtc_readers[0];
      auto ret = reader->read_l1_at(offset_idx);
      if (ret.has_value()) {
        return reader->get_current_l1_dgram();
      } else {
        /// Handle errors?
        return nullptr;
      }
    }

    std::tuple<void**,uint32_t,std::vector<uint32_t>>
    Detector::get_l1_data(size_t offset_idx,
                          const std::string& alg,
                          const std::string& data_name) {
      return (this->*get_l1_data_impl)(offset_idx, alg, data_name);
    }

    std::tuple<void**, uint32_t, std::vector<uint32_t>>
    Detector::get_l1_data_op(size_t offset_idx,
                             const std::string& alg,
                             const std::string& data_name,
                             OpFn operation) {
      return (this->*get_l1_data_op_impl)(offset_idx, alg, data_name, operation);
    }

    std::tuple<void**, uint32_t, std::vector<uint32_t>>
    Detector::get_scan_data(size_t offset_idx,
                                  const std::string& alg,
                                  const std::string& data_name) {
      // Scan will on BeginStep contain:
      // - `step_value` -- INT64
      // - `step_docstring` -- CHARSTR (maybe - always does, but not actually required)
      // - `scan_var_namexxx` -- This is the name of the variable scanned
      //   - May have multiple
      //   - E.g. `lens_h`, `lxt` etc..
      // *** EndStep will not have any data in it
      uint32_t data_rank = 0;
      std::vector<uint32_t> data_shape;
      if (!m_is_scan) {
        m_logger->warn(
            "This function is for the scan detector! "
            "Use get_l1_data/get_slow_update_data instead.");
        return std::tuple(nullptr, data_rank, data_shape);
      }

      auto& reader = m_xtc_readers[0]; // Only 1 -- The same as timing detector
      bool have_data {false};
      if (m_last_index_read == static_cast<ssize_t>(offset_idx)) {
        have_data = true;
      } else {
        std::expected<void, BDReadError> ret;
        ret = reader->read_transition_at(offset_idx, XtcData::TransitionId::BeginStep);
        have_data = ret.has_value();
      }
      if (have_data) {
        m_last_index_read = static_cast<ssize_t>(offset_idx);
        const auto& reader_seg_nos =
            reader->segment_numbers().at("scan");
        unsigned seg_no = reader_seg_nos[0]; // There should only be 1
        auto [data_ptr, data_size, rank, shape] =
          reader->get_data(m_detname, seg_no, alg, data_name);
        m_data_ptrs[seg_no] = data_ptr;
        m_data_sizes[seg_no] = data_size;
        data_rank = rank;
        data_shape = std::vector<uint32_t>(shape,shape+rank);
      } else {
        // Handle errors?
        return std::make_tuple(nullptr, data_rank, data_shape);
      }
      return std::make_tuple(m_data_ptrs.data(), data_rank, data_shape);
    }

    std::tuple<void**, uint32_t, std::vector<uint32_t>>
    Detector::get_slow_update_data(size_t offset_idx) {
      uint32_t data_rank = 0;
      std::vector<uint32_t> data_shape;
      if (!m_is_epics) {
        m_logger->warn("This function is for EPICS detectors! "
                       "Use get_l1_data/get_scan_data instead.");
        return std::make_tuple(nullptr, data_rank, data_shape);
      }
      auto& reader = m_xtc_readers[0]; // Only 1
      bool have_data {false};
      if (m_last_index_read == static_cast<ssize_t>(offset_idx)) {
        have_data = true;
      } else {
        std::expected<void, BDReadError> ret;
        ret = reader->read_transition_at(offset_idx);
        have_data = ret.has_value();
      }
      if (have_data) {
        m_last_index_read = static_cast<ssize_t>(offset_idx);
        // Data is stored under "epics" detector. The algorithm
        // is always "raw" and the field name is the PV name - our m_detname
        const auto& reader_seg_nos =
          reader->segment_numbers().at("epics");
        unsigned seg_no = reader_seg_nos[0]; // There should only be 1
        std::string epics_detname{"epics"};
        std::string epics_alg{"raw"};
        std::string pv_name{m_detname};
        auto [data_ptr, data_size, rank, shape] =
          reader->get_data(epics_detname, seg_no, epics_alg, pv_name);

        m_data_ptrs[seg_no] = data_ptr;
        m_data_sizes[seg_no] = data_size;
        data_rank = rank;
        data_shape = std::vector<uint32_t>(shape,shape+rank);
      } else {
        // Handle errors?
        return std::make_tuple(nullptr, data_rank, data_shape);
      }
      return std::make_tuple(m_data_ptrs.data(), data_rank, data_shape);
    }

    std::tuple<void**, uint32_t, std::vector<uint32_t>>
    Detector::get_l1_data_threaded(size_t offset_idx,
                                   const std::string& alg,
                                   const std::string& data_name) {
      /*
        m_logger->trace("Getting data for algorithm {} and field {} at offset idx {}",
        alg,
        data_name,
        offset_idx);
      */
      uint32_t data_rank = 0;
      std::vector<uint32_t> data_shape{0,0,0,0,0,0,0,0,0,0};

      // TODO: Setup conditional on m_last_index_read to prevent reading multiple times
      auto read_func = [&](std::shared_ptr<Base::BDReader> reader) -> void {
        auto ret = reader->read_l1_at(offset_idx);
        bool set_rank_and_shape {true};
        if (ret.has_value()) {
          const auto& reader_seg_nos =
              reader->segment_numbers().at(m_detname);
          auto seg_no_it = reader_seg_nos.begin();
          while (seg_no_it != reader_seg_nos.end()) {
            auto [data_ptr, data_size, rank, shape] =
              reader->get_data(m_detname, *seg_no_it, alg, data_name);

            m_data_ptrs[*seg_no_it] = data_ptr;
            m_data_sizes[*seg_no_it] = data_size;
            seg_no_it++;
            if (set_rank_and_shape) {
              // Set the rank to be 1 greater than per segment information
              // The first dimension will be the number of segments
              data_rank = rank + 1;
              for (size_t i=1, k=1; i <= rank; ++i) {
                if (shape[i-1] <= 1) {
                  // Flatten indices that are of size 1
                  data_rank -= 1;
                } else {
                  // Because we flatten, keep track of the index into data_shape
                  // and the index into shape separately
                  data_shape[k] = shape[i-1];
                  k++;
                }
              }
              set_rank_and_shape = false;
            }
          }
        }
      };

      std::vector<std::shared_future<void>> read_futs;
      for (auto& reader : m_xtc_readers) {
        read_futs.push_back((*m_thread_pool).enqueue(read_func, reader));
      }
      // Just wait on all the futures - we don't really care if we get stuck
      // on an early one while a later finished first. We have to wait for them
      // all
      for (auto it = read_futs.begin(); it != read_futs.end(); it++) {
        it->wait();
      }
      // Now set the first axis to be of length = number of segments
      data_shape[0] = m_data_ptrs.size();
      return std::make_tuple(m_data_ptrs.data(), data_rank, data_shape);
    }

    std::tuple<void**, uint32_t, std::vector<uint32_t>>
    Detector::get_l1_data_threaded_op(size_t offset_idx,
                                      const std::string& alg,
                                      const std::string& data_name,
                                      OpFn operation) {
      /*
        m_logger->trace("Getting data for algorithm {} and field {} at offset idx {}",
        alg,
        data_name,
        offset_idx);
      */
      uint32_t data_rank = 0;
      std::vector<uint32_t> data_shape{0,0,0,0,0,0,0,0,0,0};

      // TODO: Setup conditional on m_last_index_read to prevent reading multiple times
      auto read_func = [&](std::shared_ptr<Base::BDReader> reader) -> void {
        auto ret = reader->read_l1_at(offset_idx);
        bool set_rank_and_shape {true};
        if (ret.has_value()) {
          const auto& reader_seg_nos =
              reader->segment_numbers().at(m_detname);
          auto seg_no_it = reader_seg_nos.begin();
          while (seg_no_it != reader_seg_nos.end()) {
            auto [data_ptr, data_size, rank, shape] =
              reader->get_data(m_detname, *seg_no_it, alg, data_name);

            m_data_ptrs[*seg_no_it] = data_ptr;
            m_data_sizes[*seg_no_it] = data_size;

            /* Run the operation on the data */
            operation(det_type(),
                      *seg_no_it,
                      m_data_ptrs,
                      m_calibconst_span,
                      m_calib_data);
            seg_no_it++;
            if (set_rank_and_shape) {
              // Set the rank to be 1 greater than per segment information
              // The first dimension will be the number of segments
              data_rank = rank + 1;
              for (size_t i=1, k=1; i <= rank; ++i) {
                if (shape[i-1] <= 1) {
                  // Flatten indices that are of size 1
                  data_rank -= 1;
                } else {
                  // Because we flatten, keep track of the index into data_shape
                  // and the index into shape separately
                  data_shape[k] = shape[i-1];
                  k++;
                }
              }
              set_rank_and_shape = false;
            }
          }
        }
      };

      std::vector<std::shared_future<void>> read_futs;
      for (auto& reader : m_xtc_readers) {
        read_futs.push_back((*m_thread_pool).enqueue(read_func, reader));
      }
      // Just wait on all the futures - we don't really care if we get stuck
      // on an early one while a later finished first. We have to wait for them
      // all
      for (auto it = read_futs.begin(); it != read_futs.end(); it++) {
        it->wait();
      }
      // Now set the first axis to be of length = number of segments
      data_shape[0] = m_data_ptrs.size();
      return std::make_tuple(m_data_ptrs.data(), data_rank, data_shape);
    }


    std::tuple<void**, uint32_t, std::vector<uint32_t>>
    Detector::get_l1_data_sequential(size_t offset_idx,
                                     const std::string& alg,
                                     const std::string& data_name) {
      /*
        m_logger->trace("Getting data for algorithm {} and field {} at offset idx {}",
                        alg,
                        data_name,
                        offset_idx);
      */
      // TODO: Setup conditional on m_last_index_read to prevent reading multiple times
      // Launch all read asynchronously
      uint32_t data_rank = 0;
      std::vector<uint32_t> data_shape{0,0,0,0,0,0,0,0,0,0};
      for (auto& reader : m_xtc_readers) {
        std::expected<void, BDReadError> ret = ret = reader->iread_l1_at(offset_idx);
        if (ret.has_value()) {
          continue;
        } else {
          // Handle errors?
          return std::make_tuple(m_data_ptrs.data(), data_rank, data_shape);
        }
      }

      bool set_rank_and_shape {true};
      // Now wait on all of them
      for (auto& reader : m_xtc_readers) {
        auto ret = reader->wait();
        if (ret.has_value()) {
          //m_logger->trace("** Have a non-null dgram return. Now accessing the data field.");
          const auto& reader_seg_nos = reader->segment_numbers().at(m_detname);
          auto seg_no_it = reader_seg_nos.begin();
          while (seg_no_it != reader_seg_nos.end()) {
            auto [data_ptr, data_size, rank, shape] =
              reader->get_data(m_detname, *seg_no_it, alg, data_name);

            m_data_ptrs[*seg_no_it] = data_ptr;
            m_data_sizes[*seg_no_it] = data_size;
            if (set_rank_and_shape) {
              // Set the rank to be 1 greater than per segment information
              // The first dimension will be the number of segments
              data_rank = rank+1;
              for (size_t i=1,k=1; i<=rank; ++i) {
                if (shape[i-1] <= 1) {
                  // Flatten indices that are of size 1
                  data_rank -= 1;
                } else {
                  // Because we flatten, keep track of the index into data_shape
                  // and the index into shape separately
                  data_shape[k] = shape[i-1];
                  k++;
                }
              }
              set_rank_and_shape = false;
            }
            //m_logger->trace("*** Filled in data for segment # {}", *seg_no_it);
            seg_no_it++;
          }
        } else {
          // Handle specific errors?
          return std::make_tuple(nullptr,0,data_shape);
        }
      }
      // Now set the first axis to be of length = number of segments
      data_shape[0] = m_data_ptrs.size();
      return std::make_tuple(m_data_ptrs.data(), data_rank, data_shape);
    }

    std::tuple<void**, uint32_t, std::vector<uint32_t>>
    Detector::get_l1_data_sequential_op(size_t offset_idx,
                                        const std::string& alg,
                                        const std::string& data_name,
                                        OpFn operation) {
      /*
        m_logger->trace("Getting data for algorithm {} and field {} at offset idx {}",
                        alg,
                        data_name,
                        offset_idx);
      */
      // TODO: Setup conditional on m_last_index_read to prevent reading multiple times
      // Launch all read asynchronously
      uint32_t data_rank = 0;
      std::vector<uint32_t> data_shape{0,0,0,0,0,0,0,0,0,0};
      for (auto& reader : m_xtc_readers) {
        std::expected<void, BDReadError> ret = ret = reader->iread_l1_at(offset_idx);
        if (ret.has_value()) {
          continue;
        } else {
          // Handle errors?
          return std::make_tuple(m_data_ptrs.data(), data_rank, data_shape);
        }
      }

      bool set_rank_and_shape {true};
      // Now wait on all of them
      for (auto& reader : m_xtc_readers) {
        auto ret = reader->wait();
        if (ret.has_value()) {
          //m_logger->trace("** Have a non-null dgram return. Now accessing the data field.");
          const auto& reader_seg_nos = reader->segment_numbers().at(m_detname);
          auto seg_no_it = reader_seg_nos.begin();
          while (seg_no_it != reader_seg_nos.end()) {
            auto [data_ptr, data_size, rank, shape] =
              reader->get_data(m_detname, *seg_no_it, alg, data_name);

            m_data_ptrs[*seg_no_it] = data_ptr;
            m_data_sizes[*seg_no_it] = data_size;
            if (set_rank_and_shape) {
              // Set the rank to be 1 greater than per segment information
              // The first dimension will be the number of segments
              data_rank = rank+1;
              for (size_t i=1,k=1; i<=rank; ++i) {
                if (shape[i-1] <= 1) {
                  // Flatten indices that are of size 1
                  data_rank -= 1;
                } else {
                  // Because we flatten, keep track of the index into data_shape
                  // and the index into shape separately
                  data_shape[k] = shape[i-1];
                  k++;
                }
              }
              set_rank_and_shape = false;
            }
            // Call the operation
            operation(det_type(), *seg_no_it, m_data_ptrs, m_calibconst_span, m_calib_data);
            seg_no_it++;
          }
        } else {
          // Handle specific errors?
          return std::make_tuple(nullptr,0,data_shape);
        }
      }
      // Now set the first axis to be of length = number of segments
      data_shape[0] = m_data_ptrs.size();
      return std::make_tuple(m_data_ptrs.data(), data_rank, data_shape);
    }

  } // namespace Base
} // namespace XTCPP
