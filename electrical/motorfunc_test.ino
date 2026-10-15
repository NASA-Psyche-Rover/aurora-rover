// Nidec 24H055M020 Continuous Run Test
// Keeps the same pins as the original sketch

const int pfmPin = 9;
const int potPin = A0;

const int H1 = A1;
const int H2 = A2;
const int H3 = A3;

const int fgPin = 2;
const int dirPin = 10;
const int enPin = 11;

volatile unsigned long edgeCount = 0;

float dutyPercent = 80.0;

// Desired motor speed
unsigned long pfmFrequency = 8000;   // Hz

unsigned long previousMillis = 0;
float frequency = 0;

void countFallingEdge()
{
    edgeCount++;
}

void setup()
{
    Serial.begin(115200);

    pinMode(pfmPin, OUTPUT);

    pinMode(dirPin, OUTPUT);
    pinMode(enPin, OUTPUT);

    digitalWrite(dirPin, HIGH);   // Change to LOW for reverse
    digitalWrite(enPin, HIGH);    // Enable motor

    pinMode(fgPin, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(fgPin), countFallingEdge, FALLING);

    pinMode(H1, INPUT);
    pinMode(H2, INPUT);
    pinMode(H3, INPUT);

    // Configure Timer1 for PFM output
    TCCR1A = 0;
    TCCR1B = 0;

    TCCR1A |= (1 << COM1A1);
    TCCR1A |= (1 << WGM11);

    TCCR1B |= (1 << WGM12) | (1 << WGM13);
    TCCR1B |= (1 << CS10);

    setPWMFrequency(pfmFrequency);
}

void loop()
{
    // Keep output frequency constant
    setPWMFrequency(pfmFrequency);

    if (millis() - previousMillis >= 1000)
    {
        previousMillis = millis();

        frequency = edgeCount;
        edgeCount = 0;

        Serial.print("PFM: ");
        Serial.print(pfmFrequency);
        Serial.print(" Hz");

        Serial.print("   FG: ");
        Serial.print(frequency);
        Serial.print(" Hz");

        Serial.print("   H1: ");
        Serial.print(analogRead(H1));

        Serial.print("   H2: ");
        Serial.print(analogRead(H2));

        Serial.print("   H3: ");
        Serial.println(analogRead(H3));
    }
}

void setPWMFrequency(unsigned long frequency)
{
    if (frequency == 0)
        frequency = 1;

    unsigned long topValue = (16000000UL / frequency) - 1;

    ICR1 = topValue;
    OCR1A = (dutyPercent / 100.0) * ICR1;
}
