import numpy as np
import win32con
import win32gui
import win32ui

# Credits for https://www.youtube.com/watch?v=WymCpVUPWQ4


class WindowCapture:
    # Properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # Constructor
    def __init__(self, window_name):
        # Find the handle for the window we want to capture
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # Get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # Account for the window border and title bar and cut them off
        border_pixels = 8
        title_bar_pixels = 31
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - title_bar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = title_bar_pixels

        # Set the cropped coordinates offset so we can translate screenshot
        # Images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):
        # Get the window image data
        wdc = win32gui.GetWindowDC(self.hwnd)
        dc_obj = win32ui.CreateDCFromHandle(wdc)
        cdc = dc_obj.CreateCompatibleDC()
        data_bit_map = win32ui.CreateBitmap()
        data_bit_map.CreateCompatibleBitmap(dc_obj, self.w, self.h)
        cdc.SelectObject(data_bit_map)
        cdc.BitBlt((0, 0), (self.w, self.h), dc_obj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # Convert the raw data into a format opencv can read
        # data_bit_map.SaveBitmapFile(cdc, 'debug.bmp')
        signed_ints_array = data_bit_map.GetBitmapBits(True)
        img = np.fromstring(signed_ints_array, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # Free resources
        dc_obj.DeleteDC()
        cdc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wdc)
        win32gui.DeleteObject(data_bit_map.GetHandle())

        # Drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #  error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #  && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[..., :3]

        # Make image C_CONTIGUOUS to avoid errors that look like:
        # File ... in draw_rectangles
        # TypeError: an integer is required (got type tuple)
        # See the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img
