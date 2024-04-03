#ifndef DFF_H
#define DFF_H
#ifndef __APPLE__
  #include <halconcpp/HalconCpp.h>
#else
  #include <HALCONCpp/HalconCpp.h>
#endif

namespace DFF {

/*****************************************************************************
 * SetResourcePath
 *****************************************************************************
 * Use SetResourcePath in your application to specify the location of the 
 * HDevelop script or procedure library.
 *****************************************************************************/
  void SetResourcePath(const char* resource_path);
#ifdef _WIN32
  void SetResourcePath(const wchar_t* resource_path);
#endif

    
};

#endif