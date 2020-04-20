package bmaso.tensoralg.abstractions

import cats.free.Free

package object `integer` {
  /*
   * Coding note: A Reduce expression requires evaluator-specific representation.
   * For example, JVM class in the JVM evaluator, Aparapi-constrained class in the
   * Aparapi evaluator, or OpenCL-coding in the OpenCL evaluator. Other evaluators
   * dreamed up in the future will have their own reduction representations.
   *
   * There will be A separate algebra for each evaluator defining evaluator-specific
   * operations, such as the evaluator-specific Reduce. Then, because the Free
   * monad design makes it possible, we will combine the generic XYZTensorExprAlg
   * with evaluator-specific algebra(s) to create combo algebras. The various
   * Free monad interpreters/compilers are based on these combo algebras.
   */

  type IntTensorExpr[Eff] = Free[IntTensorExprAlg, Eff]
}
