cmake_minimum_required(VERSION 2.8.4) # Avoid moaning about cygwin
project(ai_assignment2)

set(facedir opencv_contrib-3.2.0/modules/face)

function(check_env var)
if (NOT DEFINED ${var})
  if (DEFINED ENV{${var}})
    set(${var} $ENV{${var}} CACHE PATH "Location of ${var}")
  else()
    message(SEND_ERROR "${var} is NOT defined")
    set(${var} ""           CACHE PATH "Location of ${var}")
  endif()
endif()
endfunction(check_env)

if (MSVC)
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(SEND_ERROR "Please build for 64-bit MSVS.")
  endif ()

  check_env(OPENCV_ROOT)
  include_directories(${OPENCV_ROOT}/include)
  link_directories(${OPENCV_ROOT}/x64/vc14/lib)
  set(OpenCV_LIBS opencv_world320d.lib) ### n.b. Debug: avoids a bug with getMat
else ()
  find_package(OpenCV REQUIRED)
endif (MSVC)

include_directories(${facedir}/include)
include_directories(.) # Bring in our empty cvconfig.h

add_definitions(-D__OPENCV_BUILD=1)
#add_definitions(-UHAVE_IPP)

FILE(GLOB MyLibSources ${facedir}/src/*.cpp)
add_library(libcvface ${MyLibSources})

set(prog facerec)
add_executable(${prog} ${prog}.cpp)
target_link_libraries(${prog} libcvface ${OpenCV_LIBS})

