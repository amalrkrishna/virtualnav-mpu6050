from __future__ import division
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.error import *
from OpenGL.GL import *
from PIL import Image
from scipy import integrate
from operator import add, mul, div
from copy import deepcopy
from maze_gen import NORTH, WEST, SOUTH, EAST, N, towards, isCoordinateInRange
from matplotlib.pyplot import plot, draw, show, ion

import ctypes
import copy
import sys
import math
import time
import datetime
import traceback
import urllib
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

    	
name = 'maze'

win_width, win_height = 800, 600

heading = [0, 0, 0]
loc = [0.0, 0.0, 0.0]
map_scale = 30.

keybuffer = {}
maze = []
visited = []
values = []

shader = 0
tex_wall = 0
tex_sand = 0

timer = 0
alert = 0
done = False
fps = 0
vel_x = 0
vel_y = 0
accel_x = 0
accel_y = 0
accel_xarray = []

SKYBOX_TOP      = 1
SKYBOX_BOTTOM   = 2
SKYBOX_LEFT     = 3
SKYBOX_RIGHT    = 4
SKYBOX_FRONT    = 5
SKYBOX_BACK     = 6

SKYBOX_SIZE     = 32

tex_skybox = {}
vert_skybox = {}
texc_skybox = {}

RotatingSpeed = 0.0025
MovingSpeed  = 0.005

DEBUG_COLLISON = False
DEBUG_FUNCTRACE = False
DEBUG_DRAW = False
DEBUG_AUTOPILOT = False

HUD_ALWAYS_ON = False
ENABLE_AUTOPILOT = False

def read_values():
    link = "http://10.42.0.3:8080" # Change this address to your settings
    f = urllib.urlopen(link)
    myfile = f.read()
    return myfile.split(" ")

def main():
    global heading, loc, keybuffer, timer, alert

    looper = loop()
    looper.start()
    time.sleep(4)

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(640,360)
    glutCreateWindow(name)
    glutDisplayFunc(display)
    glutReshapeFunc(resize)
    glutKeyboardFunc(keyPressed)
    glutKeyboardUpFunc(keyReleased)
    glutIdleFunc(idleFunc)

    loadShaders()
    loadTextures()
    loadSkybox()
    loadMaze()
    setupLights()

    glClearColor(0.,0.,0.,1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_CUBE_MAP)
    glDisable(GL_CULL_FACE)

    heading = [0, 0, -1]
    loc = [0.5, 0.5, -0.5]
    for i in range(256):
        keybuffer[i] = False

    timer = time.time()
    print 'timer has started.'
    glutMainLoop()
    return

def idleFunc():
    glutPostRedisplay()

def translate_maze(i):
    r = ''
    if i & NORTH:
        r += 'N'
    if i & SOUTH:
        r += 'S'
    if i & WEST:
        r += 'W'
    if i & EAST:
        r += 'E'
    return r

def loadMaze():
    global maze
    with open('maze_gen.out', 'r') as f:
        for row in f:
            maze.append(map(int, row.split(' ')))
    for i in range(N):
        visited.append([False] * N)
    for i in range(N):
        visited.append([False] * N)
    print 'first row:', map(translate_maze, maze[0])

PILOT_WALKING               = 1
PILOT_TURNLEFT              = 2
PILOT_TURNRIGHT             = 3
PILOT_REVERSE               = 4
PILOT_BACKTRACK_WALKING     = 5
PILOT_BACKTRACK_TURNLEFT    = 6
PILOT_BACKTRACK_TURNRIGHT   = 7
PILOT_COMPLETE              = 10

PILOT_ACTION_FORWARD        = 1
PILOT_ACTION_LEFT           = 2
PILOT_ACTION_RIGHT          = 4

Left = {NORTH: WEST, WEST: SOUTH, SOUTH: EAST, EAST: NORTH}
Right = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}

pilot_heading = NORTH
pilot_status = PILOT_WALKING
pilot_walked = 0.
pilot_rotated = 0.
pilot_stack = []
pilot_hint = []
pilot_stepped = 0

def translate_status(s):
    return {
        PILOT_WALKING: 'PILOT_WALKING',
        PILOT_WALKING: 'PILOT_WALKING',
        PILOT_TURNLEFT: 'PILOT_TURNLEFT',
        PILOT_TURNRIGHT: 'PILOT_TURNRIGHT',
        PILOT_REVERSE: 'PILOT_REVERSE',
        PILOT_BACKTRACK_WALKING: 'PILOT_BACKTRACK_WALKING',
        PILOT_BACKTRACK_TURNLEFT: 'PILOT_BACKTRACK_TURNLEFT',
        PILOT_BACKTRACK_TURNRIGHT: 'PILOT_BACKTRACK_TURNRIGHT',
        PILOT_COMPLETE: 'PILOT_COMPLETE'
    }[s]

def translate_action(act):
    ret = ''
    if act & PILOT_ACTION_FORWARD:
        ret += 'FORWARD '
    if act & PILOT_ACTION_LEFT:
        ret += 'LEFT '
    if act & PILOT_ACTION_RIGHT:
        ret += 'RIGHT '
    return ret

def heading_vector(d):
    return {
        NORTH: [0, 0, -1],
        SOUTH: [0, 0, 1],
        WEST: [-1, 0, 0],
        EAST: [1, 0, 0]
    }[d]


def loadTextureFromRawData(img_w, img_h, data):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    return tex

def loadTextureFromFile(fname):
    img = Image.open(fname)
    w, h = img.size
    dat = img.tostring('raw', 'RGBX', 0, -1)
    return loadTextureFromRawData(w, h, dat)

def loadSkybox():
    global tex_skybox, vert_skybox, texc_skybox

    fname = {
        SKYBOX_RIGHT: 'skybox_right.jpg',
        SKYBOX_TOP: 'skybox_top.jpg',
        SKYBOX_FRONT: 'skybox_front.jpg',
        SKYBOX_LEFT: 'skybox_left.jpg',
        SKYBOX_BOTTOM: 'skybox_bottom.jpg',
        SKYBOX_BACK: 'skybox_back.jpg'
    }

    for f,n in fname.iteritems():
        tex_skybox[f] = loadTextureFromFile(n)

    vert_skybox[SKYBOX_BACK] = [
        [SKYBOX_SIZE, SKYBOX_SIZE, SKYBOX_SIZE],
        [SKYBOX_SIZE, -SKYBOX_SIZE, SKYBOX_SIZE],
        [-SKYBOX_SIZE, SKYBOX_SIZE, SKYBOX_SIZE],
        [-SKYBOX_SIZE, -SKYBOX_SIZE, SKYBOX_SIZE]]
    texc_skybox[SKYBOX_BACK] = [[0,1], [0,0], [1,1], [1,0]]

    vert_skybox[SKYBOX_LEFT] = [
        [-SKYBOX_SIZE, SKYBOX_SIZE, SKYBOX_SIZE],
        [-SKYBOX_SIZE, -SKYBOX_SIZE, SKYBOX_SIZE],
        [-SKYBOX_SIZE, SKYBOX_SIZE, -SKYBOX_SIZE],
        [-SKYBOX_SIZE, -SKYBOX_SIZE, -SKYBOX_SIZE]]
    texc_skybox[SKYBOX_LEFT] = [[0,1], [0,0], [1,1], [1,0]]

    vert_skybox[SKYBOX_FRONT] = [
        [-SKYBOX_SIZE, SKYBOX_SIZE, -SKYBOX_SIZE],
        [-SKYBOX_SIZE, -SKYBOX_SIZE, -SKYBOX_SIZE],
        [SKYBOX_SIZE, SKYBOX_SIZE, -SKYBOX_SIZE],
        [SKYBOX_SIZE, -SKYBOX_SIZE, -SKYBOX_SIZE]]
    texc_skybox[SKYBOX_FRONT] = [[0,1], [0,0], [1,1], [1,0]]

    vert_skybox[SKYBOX_RIGHT] = [
        [SKYBOX_SIZE, SKYBOX_SIZE, -SKYBOX_SIZE],
        [SKYBOX_SIZE, -SKYBOX_SIZE, -SKYBOX_SIZE],
        [SKYBOX_SIZE, SKYBOX_SIZE, SKYBOX_SIZE],
        [SKYBOX_SIZE, -SKYBOX_SIZE, SKYBOX_SIZE]]
    texc_skybox[SKYBOX_RIGHT] = [[0,1], [0,0], [1,1], [1,0]]

    vert_skybox[SKYBOX_TOP] = [
        [SKYBOX_SIZE, SKYBOX_SIZE, SKYBOX_SIZE],
        [-SKYBOX_SIZE, SKYBOX_SIZE, SKYBOX_SIZE],
        [SKYBOX_SIZE, SKYBOX_SIZE, -SKYBOX_SIZE],
        [-SKYBOX_SIZE, SKYBOX_SIZE, -SKYBOX_SIZE]]
    texc_skybox[SKYBOX_TOP] = [[0,1], [0,0], [1,1], [1,0]]

def loadTextures():
    global tex_wall, tex_sand

    tex_wall = loadTextureFromFile('brick.jpg')
    tex_sand = loadTextureFromFile('sand.jpg')

def setupLights():
    lightZeroPosition = [10., 10., 10., 1.]
    lightZeroColor = [1.0, 1.0, 1.0, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)

def setCamera():
    global heading, loc

    if DEBUG_FUNCTRACE:
        print 'functrace: setCamera'
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    at = map(add, loc, heading)
    params = deepcopy(loc)
    params.extend(at)
    params.extend([0., 1., 0.])
    gluLookAt(*params)
    glutPostRedisplay()

def getRVcoordinates(loc):
    rel_x, rel_y = math.floor(loc[0]), math.floor(-loc[2])
    vrt_x, vrt_y = int(rel_x), int(rel_y)
    return rel_x, rel_y, vrt_x, vrt_y

def checkXBlocked(old_loc, new_loc, rx, ry, vx, vy):
    if new_loc[0] - rx < 0.2 and new_loc[0] - old_loc[0] < 0:
        if DEBUG_COLLISON:
            print 'trying reach west:',
        if maze[vx][vy] & WEST:
            if DEBUG_COLLISON:
                print 'rejected'
            new_loc[0] = rx + 0.21
        else:
            if DEBUG_COLLISON:
                print 'accepted'
        return maze[vx][vy] & WEST
    if new_loc[0] - rx > 0.8 and new_loc[0] - old_loc[0] > 0:
        if DEBUG_COLLISON:
            print 'trying reach east:',
        if maze[vx][vy] & EAST:
            if DEBUG_COLLISON:
                print 'rejected'
            new_loc[0] = rx + 0.79
        else:
            if DEBUG_COLLISON:
                print 'accepted'
        return maze[vx][vy] & EAST
    return False

def checkYBlocked(old_loc, new_loc, rx, ry, vx, vy):
    if -new_loc[2] - ry < 0.2 and -new_loc[2] - -old_loc[2] < 0:
        if DEBUG_COLLISON:
            print 'trying reach south:',
        if maze[vx][vy] & SOUTH:
            if DEBUG_COLLISON:
                print 'rejected'
            new_loc[2] = -(ry + 0.21)
        else:
            if DEBUG_COLLISON:
                print 'accepted'
        return maze[vx][vy] & SOUTH
    if -new_loc[2] - ry > 0.8 and -new_loc[2] - -old_loc[2] > 0:
        if DEBUG_COLLISON:
            print 'trying reach north:',
        if maze[vx][vy] & NORTH:
            if DEBUG_COLLISON:
                print 'rejected'
            new_loc[2] = -(ry + 0.79)
        else:
            if DEBUG_COLLISON:
                print 'accepted'
        return maze[vx][vy] & NORTH
    return False

def checkBlocked(old_loc, new_loc):
    if DEBUG_COLLISON:
        print 'testing',old_loc,'against',new_loc
    rx, ry, vx, vy = getRVcoordinates(old_loc)
    if DEBUG_COLLISON:
        print 'R', rx, ry, 'V', vx, vy
    checkXBlocked(loc, new_loc, rx, ry, vx, vy)
    checkYBlocked(loc, new_loc, rx, ry, vx, vy)

def extendsSight(i, j, d, n):
    visited[i][j] = True
    if n == 0:
        return
    if not maze[i][j] & d:
        extendsSight(*map(add, [i,j], towards[d]), d = d, n = n - 1)

class loop(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.running = 1
	def run(self):
                global values
		while True:
		    values = read_values()
	def kill(self):
		self.running = 0

accel_x_filt = []
accel_y_filt = []
gyro_z_filt = []
gyro_z_filt.append(0)
accel_x_filt.append(0)
accel_y_filt.append(0)
alpha = 0.5

def nextAnimation():
    global heading, loc, done, timer, values, accel_x, accel_y, vel_x, vel_y, alpha, accel_x_filt, accel_y_filt, gyro_z_filt

    gyro_z = int(values[0]) / 131
    accel_x = int(values[1]) * 9.8 / 16384
    accel_y = int(values[2]) * 9.8 / 16384

    with open('log.txt', 'a+') as fp:
        fp.write(str(gyro_z) + ',' + str(accel_x) + ',' + str(accel_y) + '\n')
    fp.close()
	
    gyro_z_filt.append(float(1-alpha)*float(gyro_z_filt[0]) + float(alpha * gyro_z))
    accel_x_filt.append(float(1-alpha)*float(accel_x_filt[0]) + float(alpha * accel_x))
    accel_y_filt.append(float(1-alpha)*float(accel_y_filt[0]) + float(alpha * accel_y))
	
    with open('log_filt.txt', 'a+') as fp1:
	fp1.write(str(gyro_z_filt[1]) + ',' + str(accel_x_filt[1]) + ',' + str(accel_y_filt[1]) + '\n')
    fp1.close()
    
    gyro_z_filt[0] = copy.copy(gyro_z_filt[1])
    accel_x_filt[0] = copy.copy(accel_x_filt[1])
    accel_y_filt[0] = copy.copy(accel_y_filt[1])
    
    del(gyro_z_filt[1])
    del(accel_x_filt[1])
    del(accel_y_filt[1])
	
    print "gyro_z: ", gyro_z_filt[0]
    print "accel_x: ", accel_x_filt[0]
    print "accel_y: ", accel_y_filt[0]
    
    refresh = False
    if float(gyro_z_filt[0]) > 8:
    #if keybuffer[ord('a')] and not keybuffer[ord('d')]:
        cos = math.cos(RotatingSpeed)
        sin = math.sin(RotatingSpeed)
        heading = [cos * heading[0] + sin * heading[2], heading[1], -sin * heading[0] + cos * heading[2]]
        refresh = True
    elif float(gyro_z_filt[0]) < -8:
    #elif keybuffer[ord('d')] and not keybuffer[ord('a')]:
        cos = math.cos(-RotatingSpeed)
        sin = math.sin(-RotatingSpeed)
        heading = [cos * heading[0] + sin * heading[2], heading[1], -sin * heading[0] + cos * heading[2]]
        refresh = True
    if float(accel_y_filt[0]) > 4:
    #if keybuffer[ord('w')] and not keybuffer[ord('s')]:
        new_loc = map(add, loc, map(lambda x: x * MovingSpeed, heading))
        checkBlocked(loc, new_loc)
        loc = new_loc
        refresh = True
    if float(accel_y_filt[0]) < -1.5:
    #elif keybuffer[ord('s')] and not keybuffer[ord('w')]:
        new_loc = map(add, loc, map(lambda x: x * -MovingSpeed, heading))
        checkBlocked(loc, new_loc)
        loc = new_loc
        refresh = True
    if refresh:
        rx, ry, vx, vy = getRVcoordinates(loc)
        for d in towards.iterkeys():
            extendsSight(vx,vy,d,3)
        if rx == N - 1 and ry == N - 1 and not done:
            timer = time.time() - timer
            done = True
            print 'COMPLETE, TIME ELAPSED %.2fs' % timer
        l = math.sqrt(reduce(add, map(mul, heading, heading)))
        heading = map(div, heading, [l] * 3)
        setCamera()

def resize(w, h):
    global win_width, win_height
    win_width, win_height = w, h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, float(w) / float(h) if h != 0 else float(w), 0.1, 10000)
    glMatrixMode(GL_MODELVIEW)
    setCamera()

def keyPressed(key, x, y):
    global keybuffer
    keybuffer[ord(key)] = True
    glutPostRedisplay()

def keyReleased(key, x, y):
    global keybuffer
    keybuffer[ord(key)] = False

def loadShaders():
    global shader
    vs = glCreateShader(GL_VERTEX_SHADER)
    fs = glCreateShader(GL_FRAGMENT_SHADER)

    with open('maze_pp.vs','r') as f:
        vv = f.read()
    with open('maze_pp.fs','r') as f:
        ff = f.read()
    glShaderSource(vs, vv)
    glShaderSource(fs, ff)

    glCompileShader(vs)
    if glGetShaderiv(vs, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(vs))
    glCompileShader(fs)
    if glGetShaderiv(fs, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(fs))

    shader = glCreateProgram()
    glAttachShader(shader, vs)
    glAttachShader(shader, fs)
    glLinkProgram(shader)
    glUseProgram(shader)

def drawSkyBox(x, y, w, height):
    for i in vert_skybox.iterkeys():
        glBindTexture(GL_TEXTURE_2D, tex_skybox[i])
        glBegin(GL_TRIANGLE_STRIP)
        for t, v in zip(texc_skybox[i], vert_skybox[i]):
            glTexCoord2f(*t)
            glVertex3f(*map(add, loc, v))
        glEnd()

def drawPlane(w, h):
    glBindTexture(GL_TEXTURE_2D, tex_sand)
    vert = [[w, 0, -h], [0, 0, -h], [w, 0, 0], [0, 0, 0]]
    texc = [[w, h], [0, h], [w, 0], [0, 0]]
    glBegin(GL_TRIANGLE_STRIP)
    for t, v in zip(texc, vert):
        glTexCoord2f(*t)
        glNormal3f(0., 1., 0.)
        glVertex3f(*v)
    glEnd()

def drawWallQuad(dx, dz, is_ns, negative_norm):
    vert_ns = [[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]]
    vert_we = [[0, 1, 0], [0, 1, -1], [0, 0, 0], [0, 0, -1]]
    texc = [[1, 1], [0, 1], [1, 0], [0, 0]]
    glBegin(GL_TRIANGLE_STRIP)
    for t, v in zip(texc, vert_ns if is_ns else vert_we):
        glTexCoord2f(*t)
        if is_ns:
            glNormal3f(0, 0, -1 if negative_norm else 1)
        else:
            glNormal3f(-1 if negative_norm else 1, 0, 0)
        v[0] += dx
        v[2] += dz
        glVertex3f(*v)
    glEnd()

def drawBoxWall(x, y, w):
    glBindTexture(GL_TEXTURE_2D, tex_wall)
    if w & NORTH:
        drawWallQuad(x, -y - 1, True, False)
    if w & SOUTH:
        drawWallQuad(x, -y, True, True)
    if w & WEST:
        drawWallQuad(x, -y, False, False)
    if w & EAST:
        drawWallQuad(x + 1, -y, False, True)

def drawWalls():
    pos_x, pos_y = loc[0], loc[2]
    vrt_x, vrt_y = round(pos_x), -round(pos_y)
    range_xlow, range_xhigh = vrt_x - 10, vrt_x + 10
    range_ylow, range_yhigh = vrt_y - 10, vrt_y + 10
    for i, row in enumerate(maze):
        if i > range_xlow and i < range_xhigh:
            for j, elem in enumerate(row):
                if j > range_ylow and j < range_yhigh:
                    drawBoxWall(i, j, elem & ~(WEST & SOUTH) if i != 0 and j != 0 else elem)

def glprint(x, y, str):
    glLoadIdentity()
    glColor4f(1, 0, 0, 1)
    glTranslatef(-win_width / 2, win_height / 2, 0)
    glRasterPos2f(x,-y)
    glScalef(2,2,2)
    for ch in str:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ctypes.c_int(ord(ch)))

def display():
    global timer, alert, fps
    try:
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        begin = time.time()
        glUseProgram(shader)

        nextAnimation()

        red = [1.0, 0., 0., 1.]
        green = [0., 1.0, 0., 1.]
        blue = [0., 0., 1.0, 1.]

        glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,blue)
        drawSkyBox((N - 32) / 2, (32 - N) / 2, 32, 16)
        glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,red)
        drawPlane(64, 64)
        glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,green)
        drawWalls()

        glutSwapBuffers()
        elapsed = time.time() - begin
        fps = 1. / elapsed
    except Exception as e:
        print 'error occurs:', e
        print traceback.format_exc()
        glutLeaveMainLoop()
    return

if __name__ == '__main__':
    main()
