#!/home/kaicheng/root/bin/python

import datetime
import subprocess
import StringIO
import threading
import uuid
import time
import os

homedir = os.environ['HOME']

optmap={}
optmap['blackscholes']="24 /home/cc/parsec_prebuilt/in_10M.txt black_out"
optmap['canneal']="24 80000 2000 /home/cc/parsec_prebuilt/2500000.nets 300000"
optmap['ferret']="/home/cc/parsec_prebuilt/corel/ lsh /home/cc/parsec_prebuilt/queries/ 3000 40 24 ferrer_out"
optmap['freqmine']="/home/cc/parsec_prebuilt/webdocs_250k.dat 7200"
optmap['bodytrack']="/home/cc/parsec_prebuilt/sequenceB_261/ 4 261 16000 220 0 24"
optmap['bt.C.x']="</dev/null"
optmap['cg.C.x']="</dev/null"
optmap['ft.B.x']="</dev/null"
optmap['sp.D.x']="</dev/null"
optmap['mg.B.x']="</dev/null"
optmap['ua.C.x']="</dev/null"
optmap['lu.C.x']="</dev/null"
optmap['dc.B.x']=""

# bump machines pid to a threshold to avoid pid conflict
def bumpPid(mic, num):
    last = 0
    while last < num:
        stdout = subprocess.check_output("sudo ssh %s 'true & echo $!'" % mic, shell=True)
        last = int(stdout)

def runApp(prefix, app, node, tag):
    uid = str(uuid.uuid1())
    fout = "/home/cc/exp/perf/%s-%s-%s.out" % (prefix, app, tag)
    ferr = "/home/cc/exp/err/%s-%s-%s.err" % (prefix, app, tag)
    if True:
        #print "enter cr_run"
        #print str(datetime.datetime.now())
        print "numactl --cpunodebind=%s --membind=%s /exp/prebuilt/%s </dev/null" % (node, node, app)
        stdout = subprocess.check_output("numactl --cpunodebind=%s --membind=%s /home/cc/parsec_prebuilt/%s %s 1>%s 2>%s & echo $!" % (node, node, app, optmap[app], fout, ferr), shell=True)
        print node, stdout, app
        #print str(datetime.datetime.now())
    context = {
        "app" : app,
        "mic" : node,
        "pid" : int(stdout),
        "uuid" : uid,
        "fout" : fout,
        "ferr" : ferr
    }
    return context

class MicLogger(threading.Thread):
    def __init__(self, tag, context):
        super(MicLogger, self).__init__()
        self.tag = tag
        self.context = context
        self.statfile0 = "/sys/class/xstat/stat0"
        self.statfile1 = "/sys/class/xstat/stat1"

    def run(self):
        #ts = self.context["datetime"].strftime("%Y-%m-%d-%H-%M-%S.%f")
        fmic0 = "/home/cc/exp/stats/%s-node0-stat.log" % (self.tag)
        fmic1 = "/home/cc/exp/stats/%s-node1-stat.log" % (self.tag)
        stat0 = open(fmic0, "w")
        stat1 = open(fmic1, "w")
        popen0 = subprocess.Popen(["/exp/bin/start-stat.sh", self.statfile0], stdin=None, stdout=stat0, stderr=subprocess.STDOUT)
        popen1 = subprocess.Popen(["/exp/bin/start-stat.sh", self.statfile1], stdin=None, stdout=stat1, stderr=subprocess.STDOUT)
        cd = 10
        while True:
            if not pidfinishes(self.context):
                cd = 10
            else:
               # break
                cd -= 1
                if cd < 0:
                    break
            time.sleep(1)
        #print "reach send signal"
        popen0.send_signal(9)
        popen1.send_signal(9)
        #subprocess.call("cp %s /exp/data/%s-node0-latest.log" % (fmic0, self.tag), shell=True)
        #subprocess.call("cp %s /exp/data/%s-node1-latest.log" % (fmic1, self.tag), shell=True)

def pidfinishes(context):
    #print context
    for _, v in context.iteritems():
        if isinstance(v, dict):
            #print "in pidfinish", v["pid"]
            r = subprocess.call("sudo kill -0 %d >/dev/null 2>&1" % v["pid"], shell=True)
            if r == 0:
                return False
            r = subprocess.call("sudo kill -0 %d >/dev/null 2>&1" % v["pid"], shell=True)
            if r == 0:
                return False
    return True

def runConfiguration(app1, app2, logger = True, extag = "", repeat = False):
    context = {}
    context["datetime"] = datetime.datetime.now()
    ts = context["datetime"].strftime("%Y-%m-%d-%H-%M-%S.%f")
    prefix = "%s-%s" % (app1, app2)
    c1 = runApp(prefix, app1, "0", ts)
    c2 = runApp(prefix, app2, "1", ts)
    context["app1"] = c1
    context["app2"] = c2
    if logger:
        logger = MicLogger("coolr1-1000000-%s%s-%s" % (extag, app1, app2), context)
        logger.start()
        context["logger"] = logger
    # n = open("/dev/null", "w")
    # fan = subprocess.Popen(["/mic/bin/fan-ctrl.sh", "180", "1"], stdin=None, stdout=n, stderr=subprocess.STDOUT)
    # context["fan"] = fan
    return context

def runOneNode(app, mic, logger = True, extag = "", repeat = False):
    context = {}
    context["datetime"] = datetime.datetime.now()
    ts = context["datetime"].strftime("%Y-%m-%d-%H-%M-%S.%f")
    c = runApp("", app, mic, ts)
    context["app1"] = c
    if logger:
        logger = MicLogger("coolr1-%s%s-%s" % (extag, app, "NONE"), context)
        logger.start()
        context["logger"] = logger
    # n = open("/dev/null", "w")
    # fan = subprocess.Popen(["/mic/bin/fan-ctrl.sh", "180", "1"], stdin=None, stdout=n, stderr=subprocess.STDOUT)
    # context["fan"] = fan
    return context

def stopHelper(mic, app):
    if app == "nek5":
        subprocess.call("sudo ssh %s 'killall -9 %s' >/dev/null 2>&1" % (mic, "mpiexec"), shell=True)
    elif app == "openmc":
        subprocess.call("sudo ssh %s 'killall -9 %s' >/dev/null 2>&1" % (mic, "openmc"), shell=True)
    else:
        subprocess.call("sudo killall -9 %s >/dev/null 2>&1" % (app), shell=True)

def stopContext(c):
    stopHelper(c["app1"]["mic"], c["app1"]["app"])
    if "app2" in c:
        stopHelper(c["app2"]["mic"], c["app2"]["app"])

class SwitchAppThread(threading.Thread):
    def __init__(self, context):
        super(SwitchAppThread, self).__init__()
        self.context = context

    def run(self):
        if self.context['app'] == 'nek5':
            uid = self.context["uuid"]
            subprocess.call("sudo ssh %s killall pmi_proxy" % self.context["mic"], shell=True)
            subprocess.call("sudo ssh %s killall mpiexec" % self.context["mic"], shell=True)
            nextmic = "mic1"
            if self.context["mic"] == "mic1":
                nextmic = "mic0"
            subprocess.call("sudo ssh %s '/mic/nek5/restart.sh >>%s 2>>%s'" % (nextmic, self.context['fout'], self.context['ferr']), shell=True)
            self.context["mic"] = nextmic

        elif self.context['app'] == 'openmc':
            uid = self.context["uuid"]
            subprocess.call("sudo ssh %s killall openmc" % self.context["mic"], shell=True)
            nextmic = "mic1"
            if self.context["mic"] == "mic1":
                nextmic = "mic0"
            stdout = subprocess.check_output("sudo ssh %s '/mic/openmc/restart.sh >>%s 2>>%s & echo $!'" % (nextmic, self.context['fout'], self.context['ferr']), shell=True)
            self.context["mic"] = nextmic
            self.context['pid'] = int(stdout)
        else:
            uid = self.context["uuid"]
            pid = self.context["pid"]
            dname = "%s/images/%s-%s" % (homedir, self.context['app'], uid)    
            subprocess.call("mkdir -p %s" % dname, shell=True)
            subprocess.call("sudo criu dump --tree %d --images-dir %s --shell-job" % (pid, dname), shell=True)
            #fname = "/mic/image1/%s.ckpt" % uid
            #subprocess.call("ssh %s 'cr_checkpoint --enable-NUMAware-chkpt --term -f %s -p %d'" % (self.context["mic"], fname, self.context["pid"]), shell=True)
            #subprocess.call("sudo chown yingyi:yingyi %s" % fname, shell=True)
            #subprocess.call("sudo chmod +r %s" % fname, shell=True)
            #subprocess.call("ssh %s 'cr_restart %s'" % (nextmic, fname), shell=True)
            #subprocess.call("rm -f %s" % fname, shell=True)

def switchAppAsync(context):
    th = SwitchAppThread(context)
    th.start()
    return th

def switchContextAsync(context):
    th1 = switchAppAsync(context["app1"])
    th2 = switchAppAsync(context["app2"])
    return [th1, th2]

def switchContext(context):
    thlist = switchContextAsync(context)
    for th in thlist:
        th.join()
    for c in [context["app1"], context["app2"]]: 
        pid = c["pid"]
        uid = c["uuid"]
        dname = "%s/images/%s-%s" % (homedir, c['app'], uid)
        try:
            os.kill(pid, 0)
        except OSError:
            print pid, "not exist"
        else:
            subprocess.call("killall -9 %d" % pid, shell=True)
        nextmic = "1"
        if c["mic"] == "1":
            nextmic = "0"
        subprocess.call("sudo numactl --cpunodebind=%s --membind=%s criu restore --tree %d --images-dir %s --shell-job &" % (nextmic, nextmic, pid, dname), shell=True)
        c["mic"] = nextmic

def clearStat():
    subprocess.call("sudo echo 1 > /sys/class/xstat/reset0", shell=True)
    subprocess.call("sudo echo 1 > /sys/class/xstat/reset1", shell=True)
