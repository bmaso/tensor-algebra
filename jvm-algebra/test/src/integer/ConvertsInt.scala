package bmaso.tensoralg.jvm.integer

trait ConvertsInt[T] {
  def convertIntArray(array: Array[Int]): Array[T]
  def convertInt(i: Int): T
}

object ConvertsInt {
  implicit object ConvertsIntToByte extends ConvertsInt[Byte] {
    override def convertIntArray(array: Array[Int]) = array.map(_.toByte)
    override def convertInt(i: Int) = i.toByte
  }
  implicit object ConvertsIntToShort extends ConvertsInt[Short] {
    override def convertIntArray(array: Array[Int]) = array.map(_.toShort)
    override def convertInt(i: Int) = i.toShort
  }
  implicit object ConvertsIntToInt extends ConvertsInt[Int] {
    override def convertIntArray(array: Array[Int]) = array
    override def convertInt(i: Int) = i
  }
  implicit object ConvertsIntToLong extends ConvertsInt[Long] {
    override def convertIntArray(array: Array[Int]) = array.map(_.toLong)
    override def convertInt(i: Int) = i.toLong
  }
  implicit object ConvertsIntToFloat extends ConvertsInt[Float] {
    override def convertIntArray(array: Array[Int]) = array.map(_.toFloat)
    override def convertInt(i: Int) = i.toFloat
  }
  implicit object ConvertsIntToDouble extends ConvertsInt[Double] {
    override def convertIntArray(array: Array[Int]) = array.map(_.toDouble)
    override def convertInt(i: Int) = i.toDouble
  }
}
