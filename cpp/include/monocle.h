#ifndef MONOCLE_H
#define MONOCLE_H

// Public C API for the Monocle vector search engine.
// This is the ABI boundary with Python (via ctypes). Every symbol declared
// here must have extern "C" linkage so names aren't mangled by the C++ compiler.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MonocleIndex MonocleIndex;

const char* monocle_version(void);

#ifdef __cplusplus
}
#endif

#endif
