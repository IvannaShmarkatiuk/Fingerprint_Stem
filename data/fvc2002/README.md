# FVC2002: основна вибірка

Сюди потрібно покласти зображення наборів **FVC2002 DB1_B** і **DB2_B**
(по 80 файлів .tif: 10 пальців × 8 відбитків).

1. Відкрийте сайт FVC2002 (Biometric System Laboratory, University of Bologna),
   розділ Download, і завантажте архіви `DB1_B` та `DB2_B`.
2. Розпакуйте їх у цю папку, наприклад:
   ```
   data/fvc2002/DB1_B/101_1.tif
   data/fvc2002/DB2_B/101_1.tif
   ```
3. Запуск на цій вибірці:
   ```
   python run.py --input data/fvc2002
   python experiment.py robustness --input data/fvc2002
   ```

Самі зображення до репозиторію не додаються (див. `.gitignore`).
