/**
 * Defines options for the TTGT transpose plan:
 * - measure or not plan
 * - strategy for selecting indices of transposed tensors
 */

#ifndef TTGT_OPTIMIZER_OPTIONS_H
#define TTGT_OPTIMIZER_OPTIONS_H

#include "ttgt_utils.h"
#include <exception>

class InvalidOptimizationStrategy : public std::exception
{
 private:
   const char *m_msg;

 public:
   // No-allocation constructor
   explicit InvalidOptimizationStrategy (const char *msg) noexcept
       : m_msg (msg)
   {
   }

   // Minimal overhead what()
   const char *
   what () const noexcept override
   {
      return m_msg;
   }
};

enum Strategy
{
   BASELINE // Naive strategy without optimization of the TTGT transpose
            // scheme
};

struct TTGTOptimizerOptions
{
   Strategy strategy = Strategy::BASELINE;
   bool measure_plan = false;
};

#endif
