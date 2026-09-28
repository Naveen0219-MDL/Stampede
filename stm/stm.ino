#define NUM_LEDS 12

int ledPins[NUM_LEDS] = {
  2,4,5,12,
  13,14,15,18,
  19,21,22,23
};

int ledStates[NUM_LEDS] = {0};

unsigned long prevSlow = 0;
unsigned long prevFast = 0;

bool slowState = false;
bool fastState = false;

void setup() {
  Serial.begin(115200);

  for(int i = 0; i < NUM_LEDS; i++){
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }
}

void loop() {

  // ==========================
  // SERIAL RECEIVE
  // ==========================
  if (Serial.available()) {

    String received = Serial.readStringUntil('\n');

    int index = 0;
    int start = 0;

    for (int i = 0; i < received.length(); i++) {

      if (received[i] == ',' || i == received.length() - 1) {

        String token;

        if (i == received.length() - 1)
          token = received.substring(start);
        else
          token = received.substring(start, i);

        if(index < NUM_LEDS)
          ledStates[index] = token.toInt();

        index++;
        start = i + 1;
      }
    }
  }

  unsigned long currentMillis = millis();

  // ==========================
  // GLOBAL BLINK TIMERS
  // ==========================

  if (currentMillis - prevSlow >= 1000) {
    slowState = !slowState;
    prevSlow = currentMillis;
  }

  if (currentMillis - prevFast >= 150) {
    fastState = !fastState;
    prevFast = currentMillis;
  }

  // ==========================
  // UPDATE ALL LEDs
  // ==========================

  for (int i = 0; i < NUM_LEDS; i++) {

    switch(ledStates[i]) {

      case 0: // OFF
        digitalWrite(ledPins[i], LOW);
        break;

      case 1: // SOLID
        digitalWrite(ledPins[i], HIGH);
        break;

      case 2: // SLOW BLINK
        digitalWrite(ledPins[i], slowState);
        break;

      case 3: // FAST BLINK
        digitalWrite(ledPins[i], fastState);
        break;
    }
  }
}