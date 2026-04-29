#include <Arduino.h>
#include <Servo.h>
// ============================================
// MG90S Servo Test – voller Bereich
// ESP8266 + PlatformIO
// ============================================
// platformio.ini:
// [env:nodemcuv2]
// platform = espressif8266
// board = nodemcuv2
// framework = arduino
// lib_deps = madhephaestus/ESP8266Servo @ ^1.0.0
// ============================================

#define SERVO_PIN D4

// MG90S Pulsweitenbereich (in Mikrosekunden)
// Standard:  544µs – 2400µs
// Erweitert: 400µs – 2600µs  ← mehr Bereich möglich
#define PULSE_MIN  400    // Endanschlag links  (~0°)
#define PULSE_MAX  2600   // Endanschlag rechts (~180°+)
#define DELAY_MS   12     // Geschwindigkeit (kleiner = schneller)

Servo myServo;

// Servo über Pulsweitenbereich fahren
void sweepMikros(int von, int bis, int schritt = 10) {
  if (von < bis) {
    for (int us = von; us <= bis; us += schritt) {
      myServo.writeMicroseconds(us);
      delay(DELAY_MS);
    }
  } else {
    for (int us = von; us >= bis; us -= schritt) {
      myServo.writeMicroseconds(us);
      delay(DELAY_MS);
    }
  }
}

void pause(int ms = 800) {
  delay(ms);
}

void setup() {
  Serial.begin(9600);
  delay(500);
  Serial.println("\n=== MG90S Voller Bereich Test ===");
  Serial.printf("Pulsbereich: %dµs – %dµs\n", PULSE_MIN, PULSE_MAX);

  myServo.attach(SERVO_PIN, PULSE_MIN, PULSE_MAX);

  // Mitte anfahren
  Serial.println("Startposition: Mitte");
  myServo.writeMicroseconds(1500);
  delay(1000);
}

void loop() {
  // --- Test 1: Voller Sweep ---
  Serial.println("\n[Test 1] Voller Sweep: links → rechts");
  sweepMikros(PULSE_MIN, PULSE_MAX, 10);
  pause();

  Serial.println("[Test 1] Voller Sweep: rechts → links");
  sweepMikros(PULSE_MAX, PULSE_MIN, 10);
  pause();

  // --- Test 2: Endpositionen anfahren ---
  Serial.println("\n[Test 2] Endpositionen");

  Serial.println("  → Links-Anschlag");
  myServo.writeMicroseconds(PULSE_MIN);
  pause(1000);

  Serial.println("  → Mitte (1500µs)");
  myServo.writeMicroseconds(1500);
  pause(1000);

  Serial.println("  → Rechts-Anschlag");
  myServo.writeMicroseconds(PULSE_MAX);
  pause(1000);

  // --- Test 3: Schneller Sweep (Geschwindigkeitstest) ---
  Serial.println("\n[Test 3] Schneller Sweep");
  for (int i = 0; i < 3; i++) {
    myServo.writeMicroseconds(PULSE_MIN);
    delay(300);
    myServo.writeMicroseconds(PULSE_MAX);
    delay(300);
  }

  // Zurück zur Mitte
  myServo.writeMicroseconds(1500);
  Serial.println("\nPause 3 Sekunden...");
  delay(3000);
}