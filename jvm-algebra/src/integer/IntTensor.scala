package bmaso.tensoralg.jvm.integer

import bmaso.tensoralg.abstractions.{Tensor => abstract_Tensor, Dimension}

sealed trait IntTensor extends abstract_Tensor

case class IntArrayTensor(arr: Array[Int], override val magnitude: Array[Int], offset: Int, length: Int)
    extends IntTensor {
  if(arr.length != this.elementSize) throw new IllegalArgumentException
}
