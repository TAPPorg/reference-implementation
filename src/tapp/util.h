#ifndef TAPP_UTIL_H_
#define TAPP_UTIL_H_

// This does not currently support MSVC __dllexport
#ifdef __cplusplus
#define TAPP_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define TAPP_EXPORT __attribute__((visibility("default")))
#endif

#endif

