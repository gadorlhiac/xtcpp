#ifndef XTCPP_BASE_SMDREADER_HH
#define XTCPP_BASE_SMDREADER_HH

#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/NameIndex.hh"
#include "xtcdata/xtc/ShapesData.hh" // XtcData::Name::DataType

#include "spdlog/sinks/stdout_color_sinks.h"

#include <cstdint>
#include <stdio.h>

#include <expected>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace XTCPP
{
  class DataField {
  public:
    DataField(std::string name_, XtcData::Name::DataType data_type_, unsigned rank_)
      : name(name_)
      , data_type(data_type_)
      , rank(rank_)
    {}
    std::string name;
    XtcData::Name::DataType data_type;
    unsigned rank;
  };

  /**
   * Holds information about the offset and size of a single datagram in an XTC2
   * file. A vector/array of these should be used to represent the offsets of an
   * entire XTC2 or a portion of it.
   * Note, the offset and size refer to the "big data" XTC2 files. They are however,
   * stored in the smalldata .smd.xtc2 files.
   * These offsets are for the L1Accept data only.
   */
#pragma pack(push, 1)
  struct BDXtcOffset {
    BDXtcOffset()
      : offset(0)
      , size(0)
    {}
    BDXtcOffset(uint64_t offset_, uint64_t size_)
      : offset(offset_)
      , size(size_)
    {}
    uint64_t offset; ///< Offset in CORRESPONDING big data .xtc2 file for L1Accept
    uint64_t size; ///< Size of the L1Accept datagram
  };
#pragma pack(pop)

  /**
   * Holds information about the offset and size of a single datagram in an XTC2
   * file. A vector/array of these should be used to represent the offsets of an
   * entire XTC2 or a portion of it.
   * Note, the offset and size refer to the "big data" XTC2 files. They are
   * however, stored in the smalldata .smd.xtc2 files.
   * These offsets are for the transitions only.
   */
#pragma pack(push, 1)
  struct TransitionXtcOffset {
    TransitionXtcOffset()
      : previous_l1_index(-1)
      , offset(0)
      , size(0)
      , transition_id(XtcData::TransitionId::SlowUpdate)
    {}
    TransitionXtcOffset(int64_t prev_l1_idx_,
                        uint64_t offset_,
                        uint64_t size_,
                        XtcData::TransitionId::Value transition_id_ = XtcData::TransitionId::SlowUpdate)
      : previous_l1_index(prev_l1_idx_)
      , offset(offset_)
      , size(size_)
      , transition_id(transition_id_)
    {}

    /**
     * This indicates the preivous L1Accept immediately preceeding the SlowUpdate.
     * This can be used to determine if a new SlowUpdate read is needed.
     * E.g. If previous_l1_index is 10 and event index is 11 you must read.
     */
    int64_t previous_l1_index;
    uint64_t offset; ///< Offset in CORRESPONDING big data .xtc2 file for SlowUpdate
    uint64_t size; ///< Size of the SlowUpdate datagram
    XtcData::TransitionId::Value transition_id;
  };
#pragma pack(pop)

  enum class SMDReadError {
    UnimplementedBaseFunction,
    ZeroBytesRead,
    NoOffsetInData,
    DgramHeaderError,
    PayloadTruncatedError,
    GeneralIOError
  };

  using DetAlgList = std::map<std::string, std::vector<std::string>>;
  using DetAlgDataList = std::map<std::string,
                                  std::map<std::pair<std::string,unsigned>,
                                           std::vector<DataField>>>;
  using AlgDataNameIndex = std::map<std::string, std::map<std::string, std::map<unsigned, XtcData::NameIndex>>>;
  namespace Base {
    /**
     * The SMDReader class manages reading 1 single .smd.xtc2 file.
     * It should likely only be used via a managing BDReader instance.
     */
    class SMDReader {
    public:
      SMDReader(std::string& smd_path,
                size_t max_dgram_size,
                size_t events_per_read);

      virtual ~SMDReader() {}

      /* Synchronous API */
      /**
       * Trigger a read of the .smd.xtc2 file.
       * This is a blocking call.
       */
      virtual std::expected<void, SMDReadError> read() {
        return std::unexpected(SMDReadError::UnimplementedBaseFunction);
      };

      /* Asynchronous API  */
      /**
       * Trigger an asynchronous read of the .smd.xtc2 file.
       */
      virtual std::expected<void, SMDReadError> iread() {
        return std::unexpected(SMDReadError::UnimplementedBaseFunction);
      };

      /**
       * Wait on an asynchronous read of the .smd.xtc2 file.
       * This can be called immediately after such or separated as needed.
       */
      virtual std::expected<void, SMDReadError> wait() {
        return std::unexpected(SMDReadError::UnimplementedBaseFunction);
      };

      /* Data access  */

      /**
       * Return the pointer to the current datagram.
       * This should be called after either `read` or `iread` combined with `wait`
       * @return dgram The pointer to the current datagram.
       */
      virtual XtcData::Dgram* get_current_l1_dgram() {
        return reinterpret_cast<XtcData::Dgram*>(m_access_ptr +
                                                 m_access_offset);
      }

      /**
       * Construct the offset instance into a provided buffer.
       * This should be called after having performed a read. The offset will be
       * extracted from the datagram and constructed in the buffer provided.
       *
       * An array is expected that is of size `events_per_read`. The `SMDReader`
       * instance will make sure to keep track of the number of events that have
       * been read so far so as to construct the BDXtcOffset in the correct place
       * of the external buffer's total memory.
       *
       * Note: The function also returns a pointer to the datagram/offset.
       *       this is because it uses a nullptr as a return value to indicate
       *       an error, or that there is no further offset to construct.
       *
       * @param[in] offset_buf An external buffer that the BDXtcOffset objects
       *            will be constructed into.
       * @param[in] transition_buf An external buffer that the SlowUpdate
       *            indices will be added into.
       * @return dgram The pointer to the next datagram.
       */
      XtcData::Dgram* get_offset_into(std::shared_ptr<BDXtcOffset[]> offset_buf,
                                      std::shared_ptr<TransitionXtcOffset[]> transition_buf);

      /**
       * Construct the offset instance and return it.
       * This should be called after having performed a read. The offset will be
       * extracted from the datagram.
       *
       * @param[in] external_buf An external buffer that the BDXtcOffset objects
       *            will be constructed into.
       * @return offset The offset constructed from the current datagram.
       */
      std::expected<BDXtcOffset, SMDReadError> get_offset();

      /* Getters etc - get general information */
      /**
       * Max size of a datagram (used for building buffers for reads.)
       */
      size_t max_dgram_size() const { return m_max_dgram_size; }

      /**
       * The current number of events that have been read.
       */
      size_t n_events() const { return m_curr_offset_idx; }

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
       * Contains the mapping of NameIndex objects to their fields/algs/detectors
       * to facilitate lookup at the BDReader level.
       */
      const AlgDataNameIndex&
      alg_map() const { return m_alg_map; }

      /**
       * Contains the vector algorithms per detector.
       */
      DetAlgList det_algs() const { return m_det_algs; }

      /**
       * Contains the map of data fields to algorithm per detector.
       */
      DetAlgDataList det_alg_fields() const { return m_det_alg_fields; }

      /**
       * The set of EPICS detector names (if any) in the file managed by this reader.
       */
      std::vector<std::string> epics_detnames() const { return m_epics_detnames; }

      /**
       * Whether an EndRun transition has been seen.
       */
      bool seen_end_run() const { return m_seen_end_run; }

      /**
       * The path of the .smd.xtc2 file being read.
       */
      std::string smd_path() const { return m_smd_path; }

    protected:
      virtual void init_file() {}

      void recurse_dgram_xtcs(XtcData::Xtc* xtc,
                              XtcData::TransitionId::Value transition_id);
    private:
      void extract_offset_from_dgram_into(XtcData::Xtc* xtc,
                                          std::shared_ptr<BDXtcOffset[]> external_buf);

      void inspect_xtc(XtcData::Xtc* xtc,
                       XtcData::TransitionId::Value transition_id);

    protected:
      std::string m_smd_path; ///< Path to the .smd.xtc2 file
      size_t m_events_per_read; ///< Max number of BDXtcOffset's to read at once
      size_t m_max_dgram_size; ///< Max datagram size for synchronous single reads
      size_t m_curr_offset_idx{0}; ///< Current index into BDXtcOffset buffer
      size_t m_curr_transition_index{0}; ///< Current index into TransitionXtcOffset buffer
      ssize_t m_last_l1_idx_seen{-1}; ///< Number of L1 indices passed.
      size_t m_file_offset{0}; ///< Offset in the file

      char* m_access_ptr; ///< Pointer to the buffer that has been read into
      size_t m_access_offset{0}; ///< Offset within the data buffer
      size_t m_file_size; ///< Total size of the .smd.xtc2 file

      size_t m_read_count{0}; ///< Number of bytes read on last read

      bool m_seen_end_run{false}; ///< Whether an EndRun transition has been passed

      std::vector<std::string> m_detnames;
      std::map<std::string, std::vector<unsigned>> m_segment_nos;
      std::map<std::string, std::vector<std::string>> m_serial_nos;
      std::map<std::string, std::string> m_det_types;

      std::vector<std::string> m_epics_detnames;

      /**
       * Describes how many bytes into the dgram.payload the offset information is.
       * Since both the offset and size of the L1Accept dgrams in the smd files are
       * uint64_t, the size information, which comes after the offset information,
       * is m_offset_in_payload + 8.
       * This only works for the L1Accept datagrams.
       */
      size_t m_offset_in_l1accept_payload{48};

      char* dgrams_buf; ///< Data read into this buffer

      char* m_buf;
      XtcData::NamesLookup m_names_lookup;

      std::map<std::string,                   // detname
               std::map<std::string,          // alg_name
                        std::map<unsigned,    // segment #
                                 XtcData::NameIndex>>> m_alg_map;

      DetAlgList m_det_algs; ///< Algorithm vector for each detector
      /**
       * Map of algorithms to vector of fields/type pairs for each detector
       */
      DetAlgDataList m_det_alg_fields;

      std::vector<unsigned> m_offset_in_xtc;

      std::shared_ptr<spdlog::logger> m_logger;
    };
  } // namespace Base
} // namespace XTCPP

#endif
