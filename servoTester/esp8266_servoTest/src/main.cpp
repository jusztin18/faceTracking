#include <Arduino.h>
#include <Servo.h>
// ============================================
// MG90S – 3 Servo Test
// ESP8266 + PlatformIO
// ============================================
// platformio.ini:
// [env:nodemcuv2]
// platform = espressif8266
// board = nodemcuv2
// framework = arduino
// lib_deps = madhephaestus/ESP8266Servo @ ^1.0.0
// ============================================

#include <Arduino.h>
#include <Servo.h>

// --- Pin Konfiguration ---
#define SERVO1_PIN D1
#define SERVO2_PIN D2
#define SERVO3_PIN D6

// --- MG90S Pulsweitenbereich ---
#define PULSE_MIN  500
#define PULSE_MAX  2500
#define PULSE_MID  1500
#define DELAY_MS   12

Servo servo[3];
const int pins[3] = {SERVO1_PIN, SERVO2_PIN, SERVO3_PIN};

// --- Einzelnen Servo sweepen ---
void sweepServo(int idx, int von, int bis, int schritt = 10) {
  if (von < bis) {
    for (int us = von; us <= bis; us += schritt) {
      servo[idx].writeMicroseconds(us);
      delay(DELAY_MS);
    }
  } else {
    for (int us = von; us >= bis; us -= schritt) {
      servo[idx].writeMicroseconds(us);
      delay(DELAY_MS);
    }
  }
}

// --- Alle Servos gleichzeitig auf Position ---
void alleAufPosition(int us) {
  for (int i = 0; i < 3; i++) {
    servo[i].writeMicroseconds(us);
  }
}

// --- Alle Servos gleichzeitig sweepen ---
void alleSweeepen(int von, int bis, int schritt = 10) {
  if (von < bis) {
    for (int us = von; us <= bis; us += schritt) {
      for (int i = 0; i < 3; i++) servo[i].writeMicroseconds(us);
      delay(DELAY_MS);
    }
  } else {
    for (int us = von; us >= bis; us -= schritt) {
      for (int i = 0; i < 3; i++) servo[i].writeMicroseconds(us);
      delay(DELAY_MS);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== MG90S – 3 Servo Test ===");

  for (int i = 0; i < 3; i++) {
    servo[i].attach(pins[i], PULSE_MIN, PULSE_MAX);
    servo[i].writeMicroseconds(PULSE_MID);
    Serial.printf("Servo %d: Pin D%d – bereit\n", i + 1, i + 1);
  }
  delay(1000);
}

void loop() {
  // --- Test 1: Nacheinander sweepen ---
  Serial.println("\n[Test 1] Nacheinander: jeder Servo einzeln");
  for (int i = 0; i < 3; i++) {
    Serial.printf("  Servo %d: links → rechts\n", i + 1);
    sweepServo(i, PULSE_MIN, PULSE_MAX);
    delay(300);
    sweepServo(i, PULSE_MAX, PULSE_MIN);
    delay(300);
    servo[i].writeMicroseconds(PULSE_MID);
    delay(500);
  }

  // --- Test 2: Alle gleichzeitig sweepen ---
  Serial.println("\n[Test 2] Alle gleichzeitig: links → rechts → links");
  alleSweeepen(PULSE_MIN, PULSE_MAX);
  delay(300);
  alleSweeepen(PULSE_MAX, PULSE_MIN);
  delay(500);

  // --- Test 3: Welleneffekt (Versetzt) ---
  Serial.println("\n[Test 3] Welleneffekt");
  for (int us = PULSE_MIN; us <= PULSE_MAX; us += 10) {
    servo[0].writeMicroseconds(us);
    servo[1].writeMicroseconds(constrain(us - 300, PULSE_MIN, PULSE_MAX));
    servo[2].writeMicroseconds(constrain(us - 600, PULSE_MIN, PULSE_MAX));
    delay(DELAY_MS);
  }
  for (int us = PULSE_MAX; us >= PULSE_MIN; us -= 10) {
    servo[0].writeMicroseconds(us);
    servo[1].writeMicroseconds(constrain(us + 300, PULSE_MIN, PULSE_MAX));
    servo[2].writeMicroseconds(constrain(us + 600, PULSE_MIN, PULSE_MAX));
    delay(DELAY_MS);
  }

  // --- Test 4: Endpositionen ---
  Serial.println("\n[Test 4] Endpositionen");
  Serial.println("  → Alle Links");
  alleAufPosition(PULSE_MIN);
  delay(800);

  Serial.println("  → Alle Mitte");
  alleAufPosition(PULSE_MID);
  delay(800);

  Serial.println("  → Alle Rechts");
  alleAufPosition(PULSE_MAX);
  delay(800);

  Serial.println("  → Alle Mitte");
  alleAufPosition(PULSE_MID);

  Serial.println("\nPause 3 Sekunden...");
  delay(3000);
}