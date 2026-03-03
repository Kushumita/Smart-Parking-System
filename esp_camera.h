#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
const char* ssid = "ESP32_TEST";
const char* password = "12345678";
const char* serverName = "http://192.168.43.120:5000/upload";

void startCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;

  esp_camera_init(&config);
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED)
    delay(1000);

  startCamera();
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return;

  HTTPClient http;
  http.begin(serverName);
  http.addHeader("Content-Type", "application/octet-stream");

  int httpResponseCode = http.POST(fb->buf, fb->len);
  esp_camera_fb_return(fb);

  http.end();
  delay(10000);
}