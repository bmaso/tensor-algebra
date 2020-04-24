package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import bmaso.tensoralg.abstractions.{Tensor => abstract_Tensor, Dimension}

sealed trait IntTensor extends abstract_Tensor {
  def valueAt(index: Array[Long], startingAt: Int = 0): Int
  def valueAt1D(_1dIdx: Long): Int = {
    var __1dIdx = _1dIdx
    val idx = Array.fill[Long](order)(0)
    for(d <- 0 to (order - 1);
        m = magnitude(d)) {
      idx(d) = __1dIdx % m
      __1dIdx = __1dIdx / m
    }
    valueAt(idx)
  }
}

case class IntArrayTensor(arr: Array[Int], override val magnitude: Array[Long], offset: Int)
    extends IntTensor {
  if(arr.length - offset < this.elementSize) throw new IllegalArgumentException("Backing array size is too small for elementSize")
  if(this.elementSize >= Int.MaxValue) throw new IllegalArgumentException("Magnitude is too large for this tensor implementation; elementSize must be < Int.MaxValue")

  override def valueAt(index: Array[Long], startingAt: Int = 0): Int = {
    var idx = 0L
    var multiplier = 1L
    for(ii <- startingAt to (index.length - 1)) {
      idx += index(ii) * multiplier
      multiplier *= magnitude(ii - startingAt)
    }
    arr((idx + offset).toInt)
  }
}

case class TranslateTensor(tensor: IntTensor, offsets: Array[Long])
    extends IntTensor {
  override def magnitude = tensor.magnitude
  override def valueAt(index: Array[Long], startingAt: Int = 0): Int = {
    val translatedIndex = Array.tabulate[Long](order)((n: Int) => index(startingAt + n))
    var oob = false
    for(d <- 0 to order - 1) {
      translatedIndex(d) -= offsets(d).toInt
      if(translatedIndex(d) < 0 || translatedIndex(d) >= magnitude(d)) oob = true
    }
    if(oob) 0 else tensor.valueAt(Array.copyAs[Long](translatedIndex, translatedIndex.length))
  }
}

case class BroadcastTensor(tensor: IntTensor, baseMagnitude: Array[Long])
    extends IntTensor {
  override lazy val magnitude: Array[Long] = (baseMagnitude.toList :++ tensor.magnitude.toList) toArray
  override def valueAt(index: Array[Long], startingAt: Int = 0): Int =
    tensor.valueAt(index, startingAt + baseMagnitude.length)
}

case class SliceTensor(tensor: IntTensor, sliceRange: Array[(Long, Long)])
    extends IntTensor {
  override lazy val magnitude: Array[Long] =
    sliceRange
      .map({ case (_, len) => len})
      .reverse.dropWhile(_ == 1) //...drop all final dimensions with magnitude 1...
      .reverse
  override def valueAt(index: Array[Long], startingAt: Int = 0): Int = {
    val shiftedIndex = Array.fill[Long](tensor.order)(0)
    Array.copy(index, 0, shiftedIndex, 0, index.length)
    for(ii <- 0 to (shiftedIndex.length - 1)) {
      shiftedIndex(ii) += sliceRange(ii)._1
    }
    tensor.valueAt(shiftedIndex)
  }
}
