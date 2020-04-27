package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import scala.math.max

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

/**
 * A tensor whose element values are taken from a backing array of values.
 **/
case class IntArrayTensor(arr: Array[Int], override val magnitude: Array[Long], offset: Int)
    extends IntTensor {
  if(arr.length - offset < this.elementSize) throw new IllegalArgumentException("Backing array size is too small for elementSize")
  if(this.elementSize >= Int.MaxValue) throw new IllegalArgumentException("Magnitude is too large for this tensor implementation; elementSize must be < Int.MaxValue")

  override def valueAt(index: Array[Long], startingAt: Int = 0): Int = {
    for(ii <- startingAt to index.length - 1) if(index(ii) < 0 || (index(ii) > 0 && ((ii - startingAt) >= magnitude.length || index(ii) >= magnitude(ii - startingAt))))
      throw new IllegalArgumentException("Index value is out of range")
    var idx = 0L
    var multiplier = 1L
    for(ii <- startingAt to (index.length - 1)) {
      idx += index(ii) * multiplier
      multiplier *= (if (index(ii) == 0 && (ii - startingAt) >= magnitude.length) 1L else magnitude(ii - startingAt))
    }
    arr((idx + offset).toInt)
  }
}

/**
 * A tensor "shifted" in one or more dimensions. Negative or
 * positive offsets cause the source tensor to be "truncated" in that
 * dimension.
 **/
case class TranslateTensor(tensor: IntTensor, offsets: Array[Long])
    extends IntTensor {
  /*
   * A couple odd cases to consider: offsets is longer than source magnitude,
   * offsets is shorter than source magnitude. The expected case is that
   * offsets is equal to magnitude. This code correctly handles all three cases:
   * * When offsets is longer, we want new magnitude to match and be 1 in additionally-
   *   introduced dimensions
   * * When offsets is shorter, we want to assume offsetvaue is 0 in the dimensions
   *   not referenced.
   * * When offsets are the same, we add offset to base magnitude.
   */
  override lazy val magnitude = {
    val offs = Array.fill[Long](max(tensor.order, offsets.length))(0)
    Array.copy(offsets, 0, offs, 0, offsets.length)
    val mag = offs.map(_ + 1)
    Array.copy(tensor.magnitude, 0, mag, 0, tensor.magnitude.length)
    mag
  }
  override def valueAt(index: Array[Long], startingAt: Int = 0): Int = {
    val translatedIndex = Array.tabulate[Long](order)((n: Int) => index(startingAt + n))
    var oob = false
    for(d <- 0 to order - 1) {
      translatedIndex(d) -= (if (d >= offsets.length) 0 else (offsets(d).toInt))
      if(translatedIndex(d) < 0 || translatedIndex(d) >= magnitude(d)) oob = true
    }
    if(oob) 0 else tensor.valueAt(Array.copyAs[Long](translatedIndex, translatedIndex.length))
  }
}

/**
 * A projection of an original tensor into higher dimensions by
 * duplication.
 **/
case class BroadcastTensor(tensor: IntTensor, baseMagnitude: Array[Long])
    extends IntTensor {
  override lazy val magnitude: Array[Long] = (baseMagnitude.toList :++ tensor.magnitude.toList) toArray
  override def valueAt(index: Array[Long], startingAt: Int = 0): Int =
    tensor.valueAt(index, startingAt + baseMagnitude.length)
}

/**
 * A tensor which is a "slice" of the original. The `sliceRange`
 * must provide values for all dimensions of the original tensor. The resultant
 * magnitude array is reduced in size to remove trailing "1" magnitudes.
 **/
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

/**
 * Describes a tensor the is a reindexing of the elements of another tensor.
 * The dimensionality and magnitude of this tensor may be different than the
 * original. The `elementSize` must be the same as the source tensor.
 **/
case class ReshapeTensor(tensor: IntTensor, override val magnitude: Array[Long])
    extends IntTensor {
   override def valueAt(index: Array[Long], startingAt: Int = 0): Int = {
     var idx = 0L
     var multiplier = 1L
     for(ii <- startingAt to (index.length - 1)) {
       idx += index(ii) * multiplier
       multiplier *= (if (index(ii) == 0 && (ii - startingAt) >= magnitude.length) 1L else magnitude(ii - startingAt))
     }
     tensor.valueAt1D(idx)
   }
}

/**
 * A tensor that made from the "stacking" of one or more source tensors
 * in the `joiningDimension`. The source tensors may have non-unitary size in the
 * joining dimension. Note that if the source tensors definitely have unitary
 * size in the joining dimension, then the `StackTensor` is a more efficient
 * implementation for joining -- both for in-memory size and for element value
 * access.
 **/
case class JoinTensor(tensors: Array[IntTensor], joiningDimension: Dimension)
    extends IntTensor {
  /*
   * It is assumed that the magnitude of all tensors is the same in all but
   * the joining dimension -- they may or may not be the same in the joining dimension.
   * Note the joining dimensions may be far later than the existing dimensions;
   * e.g., joining two 2x2 tensors in the _AA dimension. The following calculation
   * generates a magnitude array of the correcct size, padded with 1's as necessary,
   * whose value in the joining dimension is the sum of the backing tensors.
   */
  override lazy val magnitude = {
    val extendedTensorMags =
      for(t <- tensors) yield {
        val ex = Array.fill[Long](max(t.order, joiningDimension+1))(1)
        Array.copy(t.magnitude, 0, ex, 0, t.order)
        ex
      }
    extendedTensorMags(0)(joiningDimension) = extendedTensorMags.map(_.apply(joiningDimension)).sum
    extendedTensorMags(0)
  }
  override def valueAt(index: Array[Long], startingAt: Int = 0): Int = ???
}

/**
 * A tensor that made from the "stacking" of one or more source tensors
 * in the `joiningDimension`. The source tensors must have unitary size in the
 * joining dimension. The implementation utilizes this assumption, and is able
 * to be smaller in-memory as well as aster for element value access.

 *
 * Note there are an assumptions, which should be asserted prior to construction of an
 * instance:
 * * the magnitudes of the constituent tensors are equal in all dimensions
 * * all constituent tensors have unitary magnitude in the joining dimension
 **/
case class StackTensor(tensors: Array[IntTensor], joiningDimension: Dimension)
    extends IntTensor {
  override lazy val magnitude = {
    val extendedTensorMags =
      for(t <- tensors) yield {
        val ex = Array.fill[Long](max(t.order, joiningDimension+1))(1)
        Array.copy(t.magnitude, 0, ex, 0, t.order)
        ex
      }
    extendedTensorMags(0)(joiningDimension) = tensors.length
    extendedTensorMags(0)
  }
  override def valueAt(index: Array[Long], startingAt: Int = 0): Int = {
    val idx = Array.copyAs[Long](index, index.length)
    idx(joiningDimension + startingAt) = 0
    tensors(index((joiningDimension + startingAt).toInt).toInt).valueAt(idx, startingAt)
  }
}

/**
 * A tensor constructed from a source tensor by reversing the indexes
 * of element values along a single dimension.
 **/
 case class ReverseTensor(tensor: IntTensor, dimension: Dimension)
     extends IntTensor {
   override def magnitude = tensor.magnitude
   override def valueAt(index: Array[Long], startingAt: Int = 0): Int =
     if((index.length - startingAt) >= dimension) {
       val idx = Array.copyAs[Long](index, index.length - startingAt)
       idx(dimension) = tensor.magnitude(dimension) - 1 - idx(dimension)
       tensor.valueAt(idx, 0)
     } else
       tensor.valueAt(index)
 }

 /**
  * A tensor constructed from a source tensor by exchanging two dimensions.
  **/
 case class PivotTensor(tensor: IntTensor, dim1: Dimension, dim2: Dimension)
     extends IntTensor {
   override lazy val magnitude = {
     val mag = Array.fill[Long](max(tensor.order, max(dim1, dim2) + 1))(1)
     Array.copy(tensor.magnitude, 0, mag, 0, tensor.order)
     val swap = mag(dim1)
     mag(dim1) = mag(dim2)
     mag(dim2) = swap
     mag
   }
   override def valueAt(index: Array[Long], startingAt: Int = 0): Int = {
     if(dim1 == dim2) tensor.valueAt(index)
     else {
       val idx: Array[Long] = {
         if(index.length - startingAt > dim1 && index.length - startingAt > dim2) {
           val ret = Array.fill[Long](index.length - startingAt)(0)
           Array.copy(index, startingAt, ret, 0, index.length - startingAt)
           ret
         } else {
           val ret = Array.fill[Long](max(dim1, dim2) + 1)(0L)
           Array.copy(index, startingAt, ret, 0, index.length)
           ret
         }
       }
       val swap = idx(dim1)
       idx(dim1) = idx(dim2)
       idx(dim2) = swap
       tensor.valueAt(idx)
     }
   }
 }
