#!/bin/bash

# Load model
echo "Loading model..."
python run_model_server.py &

# Start Apache
echo "Starting Apache..."
/usr/sbin/apache2ctl -D FOREGROUND