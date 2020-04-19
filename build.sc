import mill._, scalalib._

object `abstract-algebra` extends ScalaModule {
  def scalaVersion = "2.13.1"

  def ivyDeps = Agg(
    ivy"org.typelevel::cats-free:2.1.1",
    ivy"com.softwaremill.common::tagging:2.2.1"
  )
}
