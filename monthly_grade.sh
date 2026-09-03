#!/bin/bash
# Monthly: grade the sealed forecast against the current CHS archive, rebuild the atlas + report, publish. Never edits the sealed file.
cd /home/fenexpertai/seabednet || exit 1
export OMP_NUM_THREADS=2
python3 grade_forecast.py >> grades/cron.log 2>&1 || exit 1
python3 build_atlas_v2.py >> grades/cron.log 2>&1 && python3 build_atlas_hazard.py >> grades/cron.log 2>&1 && python3 build_report.py >> grades/cron.log 2>&1 || exit 1
cd pages_repo && git pull -q && cp ../churchill_atlas_v3.html index.html && cp ../report/index.html report/index.html && mkdir -p grades && cp ../grades/latest.json grades/latest.json && cp ../grades/grade_*.json grades/ 2>/dev/null
git add -A && git -c user.name="Emilio Girard" -c user.email="girardemilio3@gmail.com" commit -q -m "Monthly forecast grade $(date +%F)" && git push -q >> ../grades/cron.log 2>&1
echo "MONTHLY_DONE $(date)" >> ../grades/cron.log
