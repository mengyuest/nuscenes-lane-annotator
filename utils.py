import numpy as np
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF, QRect
from PyQt5.QtGui import QIcon,QPainter, QBrush, QColor, QPixmap, QImage, QStandardItemModel,\
    QStandardItem, QPen
import matplotlib.pyplot as plt

def fig_to_pixmap(fig):
    buf, size = fig.canvas.print_to_buffer()
    qimage = QPixmap.fromImage(buf.tostring(), 'raw', size.width, size.height, 3)
    return QPixmap(qimage)

def remove_qimage_margin(im):
    def is_all_white(data):
        return np.min(data)>=254
    width, height = im.width(), im.height()
    buffer = im.bits().asstring(width * height * 4)  # Assuming 32-bit RGBA image
    image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))

    # Initialize crop coordinates
    left, top, right, bottom = 0, 0, width - 1, height - 1

    # Trim white borders from the left
    while left < right:
        if not is_all_white(image[:, left]):
            break
        left += 1

    # Trim white borders from the right
    while right > left:
        if not is_all_white(image[:, right]):
            break
        right -= 1

    # Trim white borders from the top
    while top < bottom:
        if not is_all_white(image[top, :]):
            break
        top += 1

    # Trim white borders from the bottom
    while bottom > top:
        if not is_all_white(image[bottom, ]):
            break
        bottom -= 1

    # Crop the image using the determined coordinates
    return im.copy(left, top, right - left + 1, bottom-top+1)

def get_lanes_nearby(nusc_map, x, y, radius):
    lanes = nusc_map.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    discrete_points = nusc_map.discretize_lanes(lanes, 0.5)
    rec_list=[]
    for lane_id, points in discrete_points.items():
        d = np.linalg.norm(np.array(points)[:, :2] - [x, y], axis=1).min()
        rec_list.append((d, lane_id, np.array(points)))
    if len(rec_list)>1:
        rec_list = sorted(rec_list, key=lambda x:x[0])
    return rec_list

def visualize_nuscenes_scene(nusc_map, ego_traj):
    r = 100
    tj_xmin = np.min(ego_traj[:,0])
    tj_xmax = np.max(ego_traj[:,0])
    tj_ymin = np.min(ego_traj[:,1]) 
    tj_ymax = np.max(ego_traj[:,1])
    patch_side_half = max(tj_xmax-tj_xmin, tj_ymax-tj_ymin)/2
    patch_center_x = (tj_xmin+tj_xmax)/2
    patch_center_y = (tj_ymin+tj_ymax)/2
    radius = r + patch_side_half
    my_patch = (patch_center_x - radius,  patch_center_y-radius, patch_center_x+radius, patch_center_y+radius)

    fig, ax = nusc_map.render_map_patch(my_patch, nusc_map.non_geometric_layers, figsize=(12, 12), bitmap=None)
    plt.plot([xxx[0]for xxx in ego_traj], [xxx[1]for xxx in ego_traj], color="blue", linestyle="--", linewidth=2)
    plt.axis("scaled")
    x_min, y_min, x_max, y_max = my_patch

    margin = 20
    ax.set_xlim(x_min-margin, x_max+margin)
    ax.set_ylim(y_min-margin, y_max+margin)
    legend = ax.get_legend()
    legend.set_visible(False)
    handles, labels = ax.get_legend_handles_labels()

    trans = ax.transData.inverted()
    fig_size_inches = fig.get_size_inches()
    dpi = fig.get_dpi()
    pixel_width = int(fig_size_inches[0] * dpi)
    pixel_height = int(fig_size_inches[1] * dpi)
    xmin, ymin = trans.transform([0, 0])
    xmax, ymax = trans.transform([pixel_width, pixel_height])

    return fig, handles, labels, xmin, xmax, ymin, ymax

def visualize_nuscenes_legends(handles, labels):
    fig = plt.figure(figsize=(2,3))
    ax2 = plt.gca()
    legend2 = ax2.legend(handles, labels)
    ax2.add_artist(legend2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_left()
    ax2.xaxis.tick_bottom()
    fig.tight_layout()
    return fig

class MyTF:
    def __init__(self, xmin, xmax, ymin, ymax, pixmap_width, pixmap_height):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.pixmap_width = pixmap_width
        self.pixmap_height = pixmap_height
        self.ratio = (xmax-xmin) / pixmap_width

    def world_to_pixel(self, x_world, y_world):
        x_pixel = (x_world - self.xmin) / self.ratio
        y_pixel = -(y_world - self.ymax) / self.ratio
        return x_pixel, y_pixel

    def pixel_to_world(self, x_pixel, y_pixel):
        x_world = x_pixel * self.ratio + self.xmin
        y_world = -y_pixel * self.ratio + self.ymax
        return x_world, y_world

class MyPainter:
    def __init__(self, qimage_cache):
        self.pixmap = QPixmap(qimage_cache)
    
    def plot_a_point(self, x, y, color, stroke):
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(color, stroke, Qt.DotLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPoint(x, y)
        painter.end()

    # TODO
    def plot_line(self, xs_pixel, ys_pixel, color=QColor(255, 0, 0, 255), stroke=1, pen=None, arrow=False):
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        if pen is None:
            pen = QPen(color, stroke, Qt.SolidLine, cap=Qt.RoundCap)
        painter.setPen(pen)
        points = [QPointF(xxx,yyy) for xxx,yyy in zip(xs_pixel, ys_pixel)]
        painter.drawPolyline(points)
        if arrow:
            angle_half = np.pi/8
            r = 16
            head = points[-1]       
            angle = np.arctan2(ys_pixel[-3]-ys_pixel[-1], xs_pixel[-3]-xs_pixel[-1])
            left_tail = QPointF(
                head.x() + r * np.cos(angle - angle_half), 
                head.y() + r * np.sin(angle - angle_half)
            )
            right_tail = QPointF(
                head.x() + r * np.cos(angle + angle_half), 
                head.y() + r * np.sin(angle + angle_half)
            )
            painter.drawPolyline([left_tail, head, right_tail])
        painter.end()

    # TODO
    def plot_a_rect(self, pts, color, stroke):
        painter = QPainter(self.pixmap)
        painter.end()

