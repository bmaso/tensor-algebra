package bmaso.tensoralg.abstractions

/**
 * Each algebra defines its own `MorphFunction` type. This trait includes the
 * common elements that must be present in the MorphFunction type.
 **/
trait MorphFunctionSignature {
  def sourceTensorMagnitude: Array[Long]
  def targetTensorMagnitude: Array[Long]
}
