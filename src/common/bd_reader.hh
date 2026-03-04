#ifndef XTCPP_BASE_BDREADER_HH
#define XTCPP_BASE_BDREADER_HH

#include "common/smd_reader.hh"

#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/TransitionId.hh"

#include "spdlog/sinks/stdout_color_sinks.h"

#include <any>
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace XTCPP {

  enum class BDReadError {
    UnimplementedBaseFunction,
    ZeroBytesRead,
    AllDgramOffsetsRead,
    GeneralIOError
  };

#pragma pack(push, 1)
  struct DataInDgramOffset {
    DataInDgramOffset(uint64_t offset_, uint64_t size_, uint32_t rank_, uint32_t* shape_)
      : offset(offset_)
      , size(size_)
      , rank(rank_)
    {
      for (size_t i=0; i<rank; ++i) {
        shape[i] = shape_[i];
      }
    }
    DataInDgramOffset()
      : offset(0)
      , size(0)
      , rank(0)
      , shape{0,0,0,0,0,0,0,0,0,0}
    {}
    uint64_t offset;
    uint64_t size;
    uint32_t rank;
    uint32_t shape[10];
  };
#pragma pack(pop)

  // A tuple of segment number, algorithm, and data field within the algorithm
  using SegAlgData = std::tuple<unsigned, std::string, std::string>;
  namespace Base {
    /**
     * The BDReader class manages reading 1 single XTC2 file.
     * It in turn manages an associated SMDReader object which will fetch the offsets
     * it can use to more efficently read through its managed XTC2 file.
     */
    class BDReader {
    public:
      BDReader(std::string& smd_path, std::string& xtc_path, size_t events_per_read);

      virtual ~BDReader();

      /* Read APIs for "big data" */
      /*****************************************************************/
      /* Synchronous API */
      /**
       * Retrieve an L1Accept datagram at an offset index.
       *
       * @param[in] offset_idx The index of the offset to use. I.e. offset index 400
       *            corresponds to the 401st event. The concrete implementations must
       *            provide the mechanism to turn these indices into the actual offset
       *            read by the SMDReader class. The index is internally wrapped
       *            by the events_per_read that was passed at creation since only
       *            this number of offsets is held in memory at a time.
       *
       * @return error May return a BDReadError with an appropriate enumerator
       *         if data is not there etc.
       */
      virtual std::expected<void, BDReadError>
      read_l1_at(size_t unwrapped_offset_idx) {
        return std::unexpected(BDReadError::UnimplementedBaseFunction);
      }

      /**
       * Retrieve a Transition datagram at an offset index.
       * This function will retrieve the most recent transition that comes BEFORE
       * the offset index which represents the event (L1Accept) number.
       *
       * @param[in] offset_idx The index of the L1Accept to use. I.e. offset index
       *            400 corresponds to the 401st event. This will be used to find
       *            the closest transition to that particular L1Accept.
       *
       * @return error May return a BDReadError with an appropriate enumerator
       *         if data is not there etc.
       */
      virtual std::expected<void, BDReadError>
      read_transition_at(size_t unwrapped_offset_idx,
                         XtcData::TransitionId::Value transition_id = XtcData::TransitionId::SlowUpdate) {
        return std::unexpected(BDReadError::UnimplementedBaseFunction);
      }

      /* Asynchronous API */
      /**
       * Trigger an asynchronous read of the datagram at the specified offset index.
       * @param[in] offset_idx The index of the offset to use. I.e. offset index 400
       *            corresponds to the 401st event. The concrete implementations must
       *            provide the mechanism to turn these indices into the actual offset
       *            read by the SMDReader class. The index is internally wrapped by
       *            the events_per_read that was passed at creation since only
       *            this number of offsets is held in memory at a time.
       *
       * @return error May return a BDReadError with an appropriate enumerator
       *         if data is not there etc.
       */
      virtual std::expected<void, BDReadError>
      iread_l1_at(size_t unwrapped_offset_idx) {
        return std::unexpected(BDReadError::UnimplementedBaseFunction);
      }

      virtual std::expected<void, BDReadError> wait() {
        return std::unexpected(BDReadError::UnimplementedBaseFunction);
      }

      /* Indirect reads via SMDReader */
      /*****************************************************************/
      /**
       * Get the next set of offsets via the managed SMDReader.
       *
       * @return num_offsets The number of offsets read. May return a
       * BDReadError if something goes wrong or no more offsets to read.
       */
      virtual std::expected<size_t, BDReadError> get_next_offsets() {
        return std::unexpected(BDReadError::UnimplementedBaseFunction);
      }

      /* Data access */
      /*****************************************************************/
      /**
       * Return the pointer to the most recently read L1Accept datagram.
       * This function must be called only after a succesful read or iread/wait.
       * The `get_data` API is generally of more interest as it actually extracts
       * the relevant components from inside the datagram.
       *
       * @return dgram A pointer to the most recently read datagram.
       */
      virtual const XtcData::Dgram* const get_current_l1_dgram() const { return nullptr; }

      /**
       * Return the pointer to the most recently read datagram.
       * This function must be called only after a succesful read or iread/wait.
       * The `get_data` API is generally of more interest as it actually extracts
       * the relevant components from inside the datagram.
       *
       * @return dgram A pointer to the most recently read datagram.
       */
      virtual const XtcData::Dgram* const get_current_transition_dgram() const {
        return nullptr;
      }

      /**
       * Return the data associated with a specific "algorithm" and field name
       * for a detector and segment number.
       * NOTE: The `read_l1_at`/`read_transition_at` function MUST be called before
       *       this one. That function reads the data from the file, this one then
       *       selects the relevant portion from within it.
       *
       * @param[in] detname The detector to get data for.
       * @param[in] seg_no The segment number for the detector.
       * @param[in] alg The algorithm, e.g. `raw`.
       * @param[in] data_name The field/data name within the algorithm. E.g. `raw`.
       * @return data_size_rank_shape The tuple of a pointer to the requested data,
       *         the size of that data in bytes, the rank and shape. The pointer
       *         may be nullptr if not found, etc.
       */
      virtual std::tuple<void*, size_t, uint32_t, uint32_t*>
      get_data(const std::string& detname,
               const unsigned& seg_no,
               const std::string& alg,
               const std::string& data_name);

      /**
       * A pointer to the offsets being used to read L1Accept datagrams.
       */
      std::shared_ptr<BDXtcOffset[]> l1_offsets() const { return m_l1_offsets; }

      /**
       * Convenience function to access the timestamp of the current datagram.
       * It is up to the caller to make sure a valid read has been performed before
       * calling this function!
       *
       * @return timestamp A 64 bit timestamp, the upper 32 bits are seconds, the lower
       *         32 bits are nanoseconds.
       */
      virtual uint64_t timestamp() const { return get_current_l1_dgram()->time.value(); }
      /**
       * Convenience function to access the seconds of the timestamp from the current
       * datagram.
       * It is up to the caller to make sure a valid read has been performed before
       * calling this function!
       *
       * @return seconds A 32 bit value representing seconds.
       */
      virtual uint32_t time_seconds() const { return get_current_l1_dgram()->time.seconds(); }
      /**
       * Convenience function to access the nanoseconds of the timestamp from the current
       * datagram.
       * It is up to the caller to make sure a valid read has been performed before
       * calling this function!
       *
       * @return seconds A 32 bit value representing nanoseconds.
       */
      virtual uint32_t time_nanoseconds() const { return get_current_l1_dgram()->time.nanoseconds(); }

      /* General information for convenience (detector names, serial numbers, etc.) */
      /*****************************************************************/
      /**
       * The set of detector names in the XTC2 file managed by this reader.
       */
      std::vector<std::string> detnames() const { return m_detnames; }

      /**
       * The map of detector names to segment numbers for the data in this XTC2 file.
       */
      const std::map<std::string, std::vector<unsigned>>& segment_numbers() const { return m_segment_nos; }

      /**
       * The map of detector names to serial numbers for the data in this XTC2 file.
       */
      std::map<std::string, std::vector<std::string>> serial_numbers() const { return m_serial_nos; }

      /**
       * The map of detector names to detector types for the data in this XTC2 file.
       */
      std::map<std::string, std::string> det_types() const { return m_det_types; }

      /**
       * The set of EPICS detector names (if any) in the file managed by this
       * reader.
       */
      std::vector<std::string> epics_detnames() const { return m_epics_detnames; }

      /**
       * Contains the vector algorithms per detector.
       * This can be used to determine which algorithms to pass to `get_data`.
       */
      DetAlgList det_algs() const { return m_det_algs; }

      /**
       * Contains the map of data fields to algorithm per detector.
       * This can be used to determine which fields to pass to `get_data`.
       */
      DetAlgDataList det_alg_fields() const { return m_det_alg_fields; }

      /* File management/information */
      /*****************************************************************/
      /**
       * Close the XTC2 file (in whatever manner appropriate for the
       * implementation). Also cleanup any additional resources.
       */
      virtual void close();

      /**
       * The path of the .smd.xtc2 file being read.
       */
      std::string smd_path() const { return m_smd_path; }

      /**
       * The path of the xtc2 file being read.
       */
      std::string xtc_path() const { return m_xtc_path; }

    protected:
      /**
       * This function is called internally by `get_data`. It provides an
       * iterative approach to pulling out the requested data if the offset
       * is not stored in `m_offsets_in_dg` or it is not reliable to use the
       * lookup.
       * In general, this function will be used only once, the first time a
       * algorithm/data field pair is requested. This function will cache the
       * offset it finds to make subsequent look-ups faster.
       *
       * @param[in] detname The detector to get data for.
       * @param[in] seg_no The segment number for the detector.
       * @param[in] alg The algorithm, e.g. `raw`.
       * @param[in] data_name The field/data name within the algorithm. E.g. `raw`.
       * @return data_size_rank_shape The tuple of a pointer to the requested data,
       *         the size of that data in bytes, the rank and shape. The pointer
       *         may be nullptr if not found, etc.
       */
      std::tuple<void*, size_t, uint32_t, uint32_t*>
      get_data_internal(const std::string& detname,
                        const unsigned& seg_no,
                        const std::string& alg,
                        const std::string& data_name);

      /**
       * Extract a value from an XTC by looking at the type/rank/size
       * information in all the auxiliary XTCs etc. See the xtcdata package for
       * more examples of this.
       *
       * @param[in] idx The index of the `Name` to lookup in the datagram.
       * @param[in] name The name object for the data being looked up.
       * @param[in] descdata The DescData constructed from the payload and a
       *            name index map - see SMDReader::alg_map for this map.
       */
      std::any get_value(size_t idx,
                         XtcData::Name& name,
                         XtcData::DescData& descdata);

      virtual void init_reader(){}; ///< Initialize the BDReader

    protected:
      size_t m_events_per_read; ///< Number of events/offsets to store in memory
      std::unique_ptr<SMDReader> m_smd_reader; ///< The SMDReader used to read .smd.xtc2
      /**
       * The buffer used to hold `m_events_per_read` offsets in memory.
       */
      std::shared_ptr<BDXtcOffset[]> m_l1_offsets{nullptr};

      /**
       * A shared buffer for holding indices for slow updates.
       * This index always points to the L1Accept index that immediately preceeds
       * a SlowUpdate. E.g. if the index is 42, that means that after the L1Accept
       * at offset index 42, there is a SlowUpdate (before you get to L1Accept at
       * the offset index of 43.). This array is signed, because a value of -1
       * indicates that before the first L1Accept, there is a SlowUpdate.
       */
      std::shared_ptr<TransitionXtcOffset[]> m_transition_offsets{nullptr};
      size_t m_curr_transition_index{0};
      size_t m_num_transitions;

      size_t m_num_events; ///< Current number of events/offsets read
      XtcData::Xtc* m_payload_ptr; ///< Pointer to the data requested by get_data
      int m_remaining_payload; ///< Remaining payload f iterating a datagram

      std::vector<std::string> m_detnames; ///< Set of detector names
      /**
       * Map of detector names to segment numbers.
       */
      std::map<std::string,std::vector<unsigned>> m_segment_nos;

      /**
       * Map of detector names to serial numbers.
       */
      std::map<std::string,std::vector<std::string>> m_serial_nos;

      /**
       * Map of detector names to detector types.
       */
      std::map<std::string,std::string> m_det_types;

      /**
       * A map of per detector offsets for fast lookup of data within a
       * datagram. This assumes consistent size of data. This is valid for
       * some algorithms but not all.
       */
      std::map<std::string, std::map<SegAlgData, DataInDgramOffset>> m_offsets_in_dg;

      std::vector<std::string> m_epics_detnames;

      std::string m_smd_path; ///< Path to the .smd.xtc2 file
      std::string m_xtc_path; ///< Path to the .xtc2 file

      std::shared_ptr<spdlog::logger> m_logger;

      DetAlgList m_det_algs;           // Detector to algorithms
      DetAlgDataList m_det_alg_fields; // Fields to algorithm, per detector
    };
  } // namespace Base
} // namespace XTCPP

#endif // XTCPP_BASE_BDREADER_HH
