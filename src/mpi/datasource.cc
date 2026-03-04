#include "datasource.hh"

#include "bd_reader.hh"
#include "detector.hh"

#include "common/detector.hh"

#include "mpi.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

namespace XTCPP {
  namespace MPI {
    DataSource::DataSource(std::string exp,
                           std::variant<std::string, int> run,
                           size_t events_per_read)
      : Base::DataSource(exp, run, events_per_read)
      , m_comm(MPI_COMM_WORLD)
    {
      MPI_Comm_rank(m_comm, &m_rank);
      MPI_Comm_size(m_comm, &m_n_ranks);
      m_local_idx = static_cast<size_t>(m_rank);
      size_t win_size{0};
      if (m_rank == 0) {
        win_size = sizeof(MPI_Aint);
      }
      MPI_Win_allocate(win_size,         /* Size */
                       sizeof(MPI_Aint), /* Displacement unit */
                       MPI_INFO_NULL,    /* Info  */
                       m_comm,           /* Communicator */
                       &m_curr_idx,      /* Base address */
                       &m_idx_window     /* Window */
      );
      if (m_rank == 0) {
        // This is okay because it gets allocated above and is not null on rank 0
        *m_curr_idx = 0;
      }
      m_offset_indices.resize(m_events_per_read);
      std::iota(m_offset_indices.begin(), m_offset_indices.end(), 0);

      if (auto tmp = spdlog::get("MPI::DataSource")) {
        m_logger = tmp;
      } else {
        m_logger = spdlog::stdout_color_mt("MPI::DataSource");
      }

      const char* fetch_idx_mode = std::getenv("XTCPP_MPIDS_IDXMODE");
      if (fetch_idx_mode && std::string(fetch_idx_mode) == "FAST") {
        fetch_next_idx_impl = &DataSource::fetch_next_idx_round_robin;
        m_logger->debug("Will use the fast deterministic mode for index distribution.");
      } else {
        fetch_next_idx_impl = &DataSource::fetch_next_idx_ordered;
        m_logger->debug("Will use the strict ordered mode for index distribution.");
      }

      init_detectors();
    }

    void DataSource::init_detectors() {
      std::string SIT_PSDM_DATA = std::getenv("SIT_PSDM_DATA");
      if (SIT_PSDM_DATA.empty()) {
        SIT_PSDM_DATA = "/sdf/data/lcls/ds";
      }
      std::string xtc_dir = SIT_PSDM_DATA + "/" + m_hutch + "/" + m_experiment + "/xtc";

      std::ostringstream oss;
      oss << std::setw(4) << std::setfill('0') << m_run;
      std::string file_base_ptn = m_experiment + "-r" + oss.str();
      for (auto const& dir_entry : fs::directory_iterator(xtc_dir)) {
        std::string xtc_path = dir_entry.path();
        if (xtc_path.find(file_base_ptn) != std::string::npos) {
          m_xtc_files.push_back(xtc_path);
          std::string xtc_stem = dir_entry.path().stem();
          std::string smd_path = xtc_dir + "/smalldata/" + xtc_stem + ".smd.xtc2";
          m_smd_files.push_back(smd_path);

          std::shared_ptr<Base::BDReader> reader = std::make_shared<BDReader>(m_comm,
                                                                              smd_path,
                                                                              xtc_path,
                                                                              m_events_per_read);
          for (const auto& detname : reader->detnames()) {
            if (m_l1_xtc_readers.find(detname) == m_l1_xtc_readers.end()) {
              m_l1_xtc_readers[detname] = {reader};
            } else {
              m_l1_xtc_readers[detname].push_back(reader);
            }
          }
          if (!reader->epics_detnames().empty()) {
            for (const auto& detname : reader->epics_detnames()) {
              if (m_epics_xtc_readers.find(detname) == m_epics_xtc_readers.end()) {
                m_epics_xtc_readers[detname] = {reader};
              }
            }
          }
        }
      }
    }

    std::shared_ptr<Base::Detector> DataSource::detector(std::string detname) {
      bool is_epics {false};
      bool is_scan {false};
      if (detname == "scan") {
        is_scan = true;
      } else if (m_l1_xtc_readers.find(detname) == m_l1_xtc_readers.end()) {
        if (m_epics_xtc_readers.find(detname) == m_epics_xtc_readers.end()) {
          throw std::runtime_error("Unknown detector type " + detname + "!");
        }
        is_epics = true;
      }
      std::vector<std::shared_ptr<Base::BDReader>> det_readers;
      if (is_scan) {
        // The scan detector will be in the same file as the timing detector
        det_readers = m_l1_xtc_readers[detname];
      } else if (is_epics) {
        det_readers = m_epics_xtc_readers[detname];
      } else {
        det_readers = m_l1_xtc_readers[detname];
      }
      std::vector<unsigned> segments;
      std::vector<std::string> serial_nos;
      std::string det_type{""};

      // Segments may not be in order so capture seg -> serial no in map
      std::map<unsigned, std::string> seg_to_serno;
      for (auto& det_reader : det_readers) {
        if (is_epics) {
          std::string ser_no = "epics1234";
          unsigned seg_no = 0;
          seg_to_serno[seg_no] = ser_no;
          det_type = "epics";
          serial_nos.push_back(ser_no);
          segments.push_back(seg_no);
        } else {
          const auto& det_reader_segs = det_reader->segment_numbers().at(detname);
          auto det_reader_sernos = det_reader->serial_numbers()[detname];
          // These two should be the same size
          if (det_reader_segs.size() != det_reader_sernos.size()) {
            throw std::runtime_error("Number of segments doesn't match number of serial nums.");
          }
          segments.insert(segments.end(),
                          det_reader_segs.begin(),
                          det_reader_segs.end());
          serial_nos.insert(serial_nos.end(),
                            det_reader_sernos.begin(),
                            det_reader_sernos.end());
          for (size_t idx = 0; idx < det_reader_segs.size(); ++idx) {
            unsigned seg_no = det_reader_segs[idx];
            std::string serno = det_reader_sernos[idx];
            seg_to_serno[seg_no] = serno;
          }
        }
        auto ret = det_reader->get_next_offsets();
        if (ret.has_value()) {
          size_t n_new_offsets = ret.value();
          if (n_new_offsets > m_last_offset_index + 1) {
            m_last_offset_index = n_new_offsets - 1;
          }

          m_xtc_readers_in_use.push_back(det_reader);
          m_logger->trace("Detector " + detname + " will read " + det_reader->xtc_path() +
                          " and " + det_reader->smd_path());
        } else {
          // Handle errors?
        }

        if (det_type.empty()) {
          det_type = det_reader->det_types()[detname];
        }
      }

      // Will need to capture the actual detector type, use placeholder for now
      std::string full_serial_no = det_type;
      for (size_t i=0; i<serial_nos.size(); ++i) {
        full_serial_no += "_" + seg_to_serno[i];
      }

      std::shared_ptr<Base::Detector> det = std::make_shared<Detector>(m_comm,
                                                                       detname,
                                                                       full_serial_no,
                                                                       segments,
                                                                       det_readers,
                                                                       m_experiment,
                                                                       m_run,
                                                                       is_epics,
                                                                       is_scan);
      return det;
    }

    size_t DataSource::fetch_next_idx() {
      return (this->*fetch_next_idx_impl)();
    }

    size_t DataSource::fetch_next_idx_ordered() {
      MPI_Aint one = 1;
      MPI_Aint curr_idx;
      MPI_Fetch_and_op(
        &one,
        &curr_idx,
        MPI_AINT,
        0,
        0,
        MPI_SUM,
        m_idx_window
      );
      return static_cast<size_t>(curr_idx);
    }

    size_t DataSource::fetch_next_idx_round_robin() {
      m_local_idx += m_n_ranks;
      return m_local_idx;
    }
  } // namespace MPI
} // namespace XTCPP

