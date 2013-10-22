#include "common/SerializationWrapper.hpp"


namespace boost {
  namespace serialization {
    // Needed to handle serialization of inf / NaN values
    namespace locales {
      std::locale default_locale(std::locale::classic(), new boost::archive::codecvt_null<char>);
      std::locale tmp_locale(default_locale, new boost::math::nonfinite_num_put<char>);
      std::locale new_locale(tmp_locale, new boost::math::nonfinite_num_get<char>);
    }
  }
};
