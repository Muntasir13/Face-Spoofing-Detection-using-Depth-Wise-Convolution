from threading import Thread, Lock
import cv2


class CameraStream(object):

    """
        Threaded class for capturing camera stream quickly in real time, media decoder: FFMPEG

        objective: to capture each frame smoothly without any delay
        output: return each frame
    """

    def __init__(self, src=0, width=800, height=600):

        """
            constructor function of class CameraStream

            input:
                src: type - int/string, source of camera. int if webcam else string if ipcam/ptz/other stream sources, default 0 for webcam
                width: type - int, camera feed width
                height: type - int, camera feed height
            
            output:
                none

        """

        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):

        """
            thread start function of class CameraStream

            input: none
            output: return to self instance

        """

        if self.started:
            print("Camera Reading Thread Already Started!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):

        """
            update thread for continuous frame reading of class CameraStream

            input: none
            output: return to self instance

        """

        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
        if self.started == False:
            return

    def read(self):

        """
            reading thread to return each frame towards driver class/function of class CameraStream

            input: none
            output: 
                ret: type - bool, if camera stream available return true, return false otherwise
                frame: type - np.array, return numpy array of each frame.
        """

        self.read_lock.acquire()
        ret = self.grabbed
        frame = self.frame.copy()
        self.read_lock.release()
        return ret, frame

    def stop(self):

        """
            thread stopper of CameraStream

            input: none
            output: none

        """

        self.started = False
        self.thread.join()
        self.stream.release()

    def __exit__(self, exc_type, exc_value, traceback):

        """
            destroyer of the thread class of CameraStream
        """
        self.stream.release()

class GstStream(object):

    """
        Threaded class for capturing camera stream quickly in real time, media decoder: Gstreamer

        objective: to capture each frame smoothly without any delay
        output: return each frame
    """

    def __init__(self, src=0, width=800, height=600):

        """
            constructor function of class CameraStream

            input:
                src: type - int/string, source of camera. int if webcam else string if ipcam/ptz/other stream sources, default 0 for webcam
                width: type - int, camera feed width
                height: type - int, camera feed height
            
            output:
                none

        """

        self.stream = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):

        """
            thread start function of class CameraStream

            input: none
            output: return to self instance

        """

        if self.started:
            print("Camera Reading Thread Already Started!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):

        """
            update thread for continuous frame reading of class CameraStream

            input: none
            output: return to self instance

        """

        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
        if self.started == False:
            return

    def read(self):

        """
            reading thread to return each frame towards driver class/function of class CameraStream

            input: none
            output: 
                ret: type - bool, if camera stream available return true, return false otherwise
                frame: type - np.array, return numpy array of each frame.
        """

        self.read_lock.acquire()
        ret = self.grabbed
        frame = self.frame.copy()
        self.read_lock.release()
        return ret, frame

    def stop(self):

        """
            thread stopper of CameraStream

            input: none
            output: none

        """

        self.started = False
        self.thread.join()
        self.stream.release()

    def __exit__(self, exc_type, exc_value, traceback):

        """
            destroyer of the thread class of CameraStream
        """

        self.stream.release()