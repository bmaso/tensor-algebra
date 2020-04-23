package bmaso.tensoralg.abstractions

/**
 * A `Tensor` object represents the native representation of the tensor's
 * data at runtime. ou cannot access the element values of a `Tensor` through
 * a `Tensor`. Rather, you use each runtime's definition of `TensorExprExecutable`
 * to perform one of two operations:
 * * create a `Tensor` from some Scala-native data structure, such as an array
 *   or NIO buffer
 * * copy `Tensor` values to a Scala-native data structure (again, examples
 *   include an array or NIO buffer)
 *
 * The only data a `Tensor` object carries with it is the `magnitude`. The
 * int values in the `magnitude` array indicate the individual dimension
 * magnitudes of the tensor.
 *
 * The length of the `magnitude` array indicates the tensor's order. That is,
 * 1-D, 3-D, etc.
 *
 * The total numberof elements in the tensor is the `elementSize`. This is
 * computed as the product of the magntiude array.
 **/
abstract class Tensor {
  def magnitude: Array[Long]
  def order = magnitude.length
  val elementSize: Long = if (magnitude.isEmpty) 0 else magnitude.reduce(_ * _)
}
