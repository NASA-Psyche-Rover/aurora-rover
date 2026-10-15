#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55);

void setup(void)
{
  Serial.begin(9600);

  Serial.println("BNO055 Orientation + Angular Velocity Test");
  Serial.println("");

  // Initialize the BNO055
  if (!bno.begin())
  {
    Serial.println("Ooops, no BNO055 detected...");
    Serial.println("Check your wiring or I2C address!");

    while (1);
  }

  delay(1000);

  // Use the external crystal on the BNO055
  bno.setExtCrystalUse(true);
}


void loop(void)
{

  imu::Vector<3> orientation =
      bno.getVector(Adafruit_BNO055::VECTOR_EULER);

  imu::Vector<3> gyro =
      bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

  Serial.print("OX: ");
  Serial.print(orientation.x(), 4);

  Serial.print("\tOY: ");
  Serial.print(orientation.y(), 4);

  Serial.print("\tOZ: ");
  Serial.print(orientation.z(), 4);

  Serial.print("\tAVX: ");
  Serial.print(gyro.x(), 4);

  Serial.print("\tAVY: ");
  Serial.print(gyro.y(), 4);

  Serial.print("\tAVZ: ");
  Serial.print(gyro.z(), 4);

  Serial.println();

  // 20 readings per second
  delay(50);
}