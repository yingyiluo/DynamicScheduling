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
    #time.sleep(60)
    libcool.clearStat()
    logging.info("Running %s %s", sys.argv[1], sys.argv[2])
    start = time.time()
    c = libcool.runConfiguration(sys.argv[1], sys.argv[2])
    time.sleep((25 * 60) - (time.time() - start))
    libcool.stopContext(c)
    c["logger"].join()
    logging.info("Finished %s %s", sys.argv[1], sys.argv[2])

if False:
    time.sleep(60)
    libcool.clearStat()
    logging.info("Running %s %s", sys.argv[2], sys.argv[1])
    start = time.time()  
    c = libcool.runConfiguration(sys.argv[2], sys.argv[1])
    time.sleep((20 * 60) - (time.time() - start))
    libcool.stopContext(c)
    c["logger"].join()
    logging.info("Finished %s %s", sys.argv[2], sys.argv[1])

if False:
    time.sleep(60)
    libcool.clearStat()
    logging.info("Running %s %s, switch at 10min", sys.argv[1], sys.argv[2])
    start = time.time()
    c = libcool.runConfiguration(sys.argv[1], sys.argv[2], extag = "switch-")
    time.sleep((10 * 60) - (time.time() - start))
    start = time.time()
    libcool.switchContext(c)
    time.sleep((50 * 60) - (time.time() - start))
    libcool.stopContext(c)
    c["logger"].join()
    logging.info("Finished %s %s", sys.argv[1], sys.argv[2])

    time.sleep(60)
    libcool.clearStat()
    logging.info("Running %s %s, switch at 10min", sys.argv[2], sys.argv[1])
    start = time.time()
    c = libcool.runConfiguration(sys.argv[2], sys.argv[1], extag = "switch-")
    time.sleep((10 * 60) - (time.time() - start))
    start = time.time()
    libcool.switchContext(c)
    time.sleep((50 * 60) - (time.time() -start))
    libcool.stopContext(c)
    c["logger"].join()
    logging.info("Finished %s %s", sys.argv[2], sys.argv[1])

logging.info("exp1 completed")
sys.exit(0)
