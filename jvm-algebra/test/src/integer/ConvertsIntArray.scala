package bmaso.tensoralg.jvm.integer

trait ConvertsIntArray[T] {
  def convertIntArray(array: Array[Int]): Array[T]
}

object ConvertsIntArray {
  implicit object ConvertsIntArrayToByte extends ConvertsIntArray[Byte] {
    override def convertIntArray(array: Array[Int]) = array.map(_.toByte)
  }
  implicit object ConvertsIntArrayToShort extends ConvertsIntArray[Short] {
    override def convertIntArray(array: Array[Int]) = array.map(_.toShort)
  }
  implicit object ConvertsIntArrayToInt extends ConvertsIntArray[Int] {
    override def convertIntArray(array: Array[Int]) = array
  }
  implicit object ConvertsIntArrayToLong extends ConvertsIntArray[Long] {
    override def convertIntArray(array: Array[Int]) = array.map(_.toLong)
  }
}
