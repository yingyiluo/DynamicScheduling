#!/home/kaicheng/root/bin/python

import libcool
import sys
import time
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename="exp1.log", level=logging.INFO)

logging.info("Starting new exp.")

if len(sys.argv) != 3:
    print "program app0 app1"
    sys.exit(1)


if True:
    time.sleep(60)
    libcool.clearStat()
    logging.info("Running %s %s", sys.argv[1], sys.argv[2])
    c = libcool.runConfiguration(sys.argv[1], sys.argv[2])
    time.sleep(60 * 60)
    libcool.stopContext(c)
    c["logger"].join()
    logging.info("Finished %s %s", sys.argv[1], sys.argv[2])

    time.sleep(60)
    libcool.clearStat()
    logging.info("Running %s %s", sys.argv[2], sys.argv[1])
    c = libcool.runConfiguration(sys.argv[2], sys.argv[1])
    time.sleep(60 * 60)
    libcool.stopContext(c)
    c["logger"].join()
    logging.info("Finished %s %s", sys.argv[2], sys.argv[1])

if False:
    libcool.setCool()
    time.sleep(60)
    libcool.clearStat()
    libcool.setHot()
    logging.info("Running %s %s, switch at 10min", sys.argv[1], sys.argv[2])
    c = libcool.runConfiguration(sys.argv[1], sys.argv[2], extag = "switch-")
    time.sleep(10 * 60)
    libcool.switchContext(c)
    time.sleep(50 * 60)
    libcool.stopContext(c)
    c["logger"].join()
    logging.info("Finished %s %s", sys.argv[1], sys.argv[2])

    libcool.setCool()
    time.sleep(60)
    libcool.clearStat()
    libcool.setHot()
    logging.info("Running %s %s, switch at 10min", sys.argv[2], sys.argv[1])
    c = libcool.runConfiguration(sys.argv[2], sys.argv[1], extag = "switch-")
    time.sleep(10 * 60)
    libcool.switchContext(c)
    time.sleep(50 * 60)
    libcool.stopContext(c)
    c["logger"].join()
    logging.info("Finished %s %s", sys.argv[2], sys.argv[1])

logging.info("exp1 completed")
sys.exit(0)
