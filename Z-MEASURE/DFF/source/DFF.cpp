#include "DFF.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <string>

#ifndef __APPLE__
  #include <hdevengine/HDevEngineCpp.h>
#else
  #include <HDevEngineCpp/HDevEngineCpp.h>
#endif

using namespace HalconCpp;
using namespace HDevEngineCpp;

namespace DFF 
{
  std::string sgResourcePath;
  bool AddResourcePathToProcedurePath()
  {
    HDevEngineCpp::HDevEngine().AddProcedurePath(sgResourcePath.c_str());
    return true;
  }
  bool LazyInitProcedurePath()
  {
    static std::mutex lock;
    std::unique_lock<std::mutex> locker(lock);
    static const bool init = AddResourcePathToProcedurePath();
    return init;
  }
  void SetResourcePath(const char* resource_path)
  {
    sgResourcePath = resource_path;
    std::replace(sgResourcePath.begin(),sgResourcePath.end(), '\\','/');
    if(sgResourcePath.length() > 0 && sgResourcePath[sgResourcePath.length()-1]!='/')
    {
      sgResourcePath+="/";
    }
    AddResourcePathToProcedurePath();
  }

#ifdef _WIN32
  void SetResourcePath(const wchar_t* resource_path)
  {
    SetResourcePath(resource_path ? HString(resource_path).TextA() : NULL);
  }
#endif
  template <typename T>
  struct ParamHandler
  {
  };
  template <>
  struct ParamHandler<HalconCpp::HTuple>
  {
    static void SetParameter(HDevEngineCpp::HDevProcedureCall& proc,
        const char*                                     name,
        HalconCpp::HTuple const&                        parameter)
    {
      proc.SetInputCtrlParamTuple(name, parameter);
    }
    static HalconCpp::HTuple GetParameter(
        HDevEngineCpp::HDevProcedureCall& proc, const char* name)
    {
      return proc.GetOutputCtrlParamTuple(name);
    }
  };
  template <>
  struct ParamHandler<HalconCpp::HObject>
  {
    static void SetParameter(HDevEngineCpp::HDevProcedureCall& proc,
        const char*                                     name,
        HalconCpp::HObject const&                       parameter)
    {
      proc.SetInputIconicParamObject(name, parameter);
    }

    static HalconCpp::HObject GetParameter(
        HDevEngineCpp::HDevProcedureCall& proc, const char* name)
    {
      return proc.GetOutputIconicParamObject(name);
    }
  };
  template <>
  struct ParamHandler<HalconCpp::HTupleVector>
  {
    static void SetParameter(HDevEngineCpp::HDevProcedureCall& proc,
        const char*                                     name,
        HalconCpp::HTupleVector const&                  parameter)
    {
      proc.SetInputCtrlParamVector(name, parameter);
    }

    static HalconCpp::HTupleVector GetParameter(
        HDevEngineCpp::HDevProcedureCall& proc, const char* name)
    {
      return proc.GetOutputCtrlParamVector(name);
    }
  };
  template <>
  struct ParamHandler<HalconCpp::HObjectVector>
  {
    static void SetParameter(HDevEngineCpp::HDevProcedureCall& proc,
        const char*                                     name,
        HalconCpp::HObjectVector const&                 parameter)
    {
      proc.SetInputIconicParamVector(name, parameter);
    }

    static HalconCpp::HObjectVector GetParameter(
        HDevEngineCpp::HDevProcedureCall& proc, const char* name)
    {
      return proc.GetOutputIconicParamVector(name);
    }
  };

  HDevProgram GetProgram(std::string const& program_file)
  {
    static std::mutex lock;
    static std::map<std::string,HDevProgram> programs;

    std::unique_lock<std::mutex> locker(lock);

    auto prog_iter = programs.find(program_file);
    if(prog_iter != programs.end())
    {
      return prog_iter->second;
    }
    else
    {
      HDevProgram program(program_file.c_str());
      programs[program_file] = program;
      return program;
    }
    return HDevProgram();
  }
};