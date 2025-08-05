#ifndef HI_TAPP_ERROR_H_
#define HI_TAPP_ERROR_H_

#include <stddef.h>
#include <stdbool.h>

typedef int HI_TAPP_error;

/* Return true if the error code indicates success and false otherwise. */
bool HI_TAPP_check_success(HI_TAPP_error error);

/*
 * Fill a user-provided buffer with an implementation-defined string explaining the error code. No more than maxlen-1
 * characters will be written. If maxlen is greater than zero, then a terminating null character is also
 * written. The actual number of characters written is returned, not including the terminating null character.
 * If maxlen is zero, then no characters are written and instead the length of the full string which would have been
 * written is returned, not including the terminating null character. This means that the message written will always
 * be null-terminated.
 *
 * TODO: should the null character be included in the return value?
 */
size_t HI_TAPP_explain_error(HI_TAPP_error error,
                          size_t maxlen,
                          char* message);

#endif /* HI_TAPP_ERROR_H_ */
