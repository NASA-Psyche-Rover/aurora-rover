//Rough code to control servo motors to drive pulleys

#include <ESP32Servo.h> // includes servo library

Servo myservo1;  // create servo object to control a servo
Servo myservo2;
Search


int pos1 = 0; // variable to store the servo position in degrees
int pos2 = 0;

int servpin1 = ; //GPIO pins
int servpin2 = ;

string loc = "XX";

void setup() {
  
  myservo1.attach();
  myservo2.attach();

  myservo1.write(pos1 = 0); // make sure servo pos starts at 0 degs
  myservo2.write(pos2 = 0);
  delay(1000);

}

void loop() {
  cout << "Set position: " << endl;
  cin >> loc;
  
}
