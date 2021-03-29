import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

def closest_node(xy, pts):
    #search the list of nodes for the one closest to node, return the name
    dist_2 = np.sqrt(np.sum((pts - np.array(xy).reshape((-1, 2)))**2, axis=1))
    if (dist_2[np.argmin(dist_2)] > 20):
        return -1
    return np.argmin(dist_2)


def draw_landmarks(img, pts, pc=(0,0,255), radius=2, lc=(0,255,0), thickness=2):

    for i in range(0, 16):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (0, 255, 0), thickness)
    for i in range(17, 21):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 0, 0), thickness)
    for i in range(22, 26):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 0, 0), thickness)
    for i in range(27, 35):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 255, 0), thickness)
    for i in range(36, 41):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 0, 255), thickness)
    for i in range(42, 47):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 0, 255), thickness)
    for i in range(48, 59):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 128, 0), thickness)
    for i in range(60, 67):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 128, 128), thickness)
    cv2.line(img, (int(pts[48, 0]), int(pts[48, 1])),
             (int(pts[59, 0]), int(pts[59, 1])), (255, 128, 0), thickness)
    cv2.line(img, (int(pts[60, 0]), int(pts[60, 1])),
             (int(pts[67, 0]), int(pts[67, 1])), (255, 128, 128), thickness)

    for i in range(68):
        cv2.circle(img, (int(pts[i, 0]), int(pts[i, 1])), radius, pc, -1)


def norm_anno(ROOT_DIR, CH, param=[0.75, 0.35, 0.6, 0.6], show=True):

    face_tmp = np.loadtxt(os.path.join(ROOT_DIR, CH + '_face_open_mouth.txt'))  # .reshape(1, 204)
    try:
        face_tmp = face_tmp.reshape(68, 3)
    except:
        print('annotated face is not in correct size = [68 x 3]')
        exit(0)

    scale = 1.6 / (face_tmp[0, 0] - face_tmp[16, 0])
    shift = - 0.5 * (face_tmp[0, 0:2] + face_tmp[16, 0:2])
    face_tmp[:, 0:2] = (face_tmp[:, 0:2] + shift) * scale
    face_std = np.loadtxt(os.path.join(ROOT_DIR, 'STD_FACE_LANDMARKS.txt'))
    face_std = face_std.reshape(68, 3)

    face_tmp[:, -1] = face_std[:, -1]
    face_tmp[:, 0:2] = -face_tmp[:, 0:2]
    np.savetxt(os.path.join(ROOT_DIR, CH + '_face_open_mouth_norm.txt'), face_tmp, fmt='%.4f')
    np.savetxt(os.path.join(ROOT_DIR, CH + '_scale_shift.txt'), np.array([scale, shift[0], shift[1]]), fmt='%.10f')

    # Force the frame to close mouth
    face_tmp[49:54, 1] = param[0] * face_tmp[49:54, 1] + (1-param[0]) * face_tmp[59:54:-1, 1]
    face_tmp[59:54:-1, 1] = param[1] * face_tmp[49:54, 1] + (1-param[1]) * face_tmp[59:54:-1, 1]
    face_tmp[61:64, 1] = param[2] * face_tmp[61:64, 1] + (1-param[2]) * face_tmp[67:64:-1, 1]
    face_tmp[67:64:-1, 1] = param[3] * face_tmp[61:64, 1] + (1-param[3]) * face_tmp[67:64:-1, 1]
    face_tmp[61:64, 0] = 0.6 * face_tmp[61:64, 0] + 0.4 * face_tmp[67:64:-1, 0]
    face_tmp[67:64:-1, 0] = 0.6 * face_tmp[61:64, 0] + 0.4 * face_tmp[67:64:-1, 0]

    np.savetxt(os.path.join(ROOT_DIR, CH + '_face_close_mouth.txt'), face_tmp, fmt='%.4f')

    std_face_id = np.loadtxt(os.path.join(ROOT_DIR, CH + '_face_close_mouth.txt'))  # .reshape(1, 204)
    std_face_id = std_face_id.reshape(68, 3)

    def vis_landmark_on_plt(fl, x_offset=0.0, show_now=True):
        def draw_curve(shape, idx_list, loop=False, x_offset=0.0, c=None):
            for i in idx_list:
                plt.plot((shape[i, 0] + x_offset, shape[i + 1, 0] + x_offset), (-shape[i, 1], -shape[i + 1, 1]), c=c)
            if (loop):
                plt.plot((shape[idx_list[0], 0] + x_offset, shape[idx_list[-1] + 1, 0] + x_offset),
                         (-shape[idx_list[0], 1], -shape[idx_list[-1] + 1, 1]), c=c)

        draw_curve(fl, list(range(0, 16)), x_offset=x_offset)  # jaw
        draw_curve(fl, list(range(17, 21)), x_offset=x_offset)  # eye brow
        draw_curve(fl, list(range(22, 26)), x_offset=x_offset)
        draw_curve(fl, list(range(27, 35)), x_offset=x_offset)  # nose
        draw_curve(fl, list(range(36, 41)), loop=True, x_offset=x_offset)  # eyes
        draw_curve(fl, list(range(42, 47)), loop=True, x_offset=x_offset)
        draw_curve(fl, list(range(48, 59)), loop=True, x_offset=x_offset, c='b')  # mouth
        draw_curve(fl, list(range(60, 67)), loop=True, x_offset=x_offset, c='r')
        draw_curve(fl, list(range(60, 64)), loop=False, x_offset=x_offset, c='g')

        if (show_now):
            plt.show()

    vis_landmark_on_plt(std_face_id, show_now=show)



# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


# Draw voronoi diagram
def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), -1, cv2.LINE_AA, 0)
    print("end of draw_voronoi")


def delauney_tri(ROOT_DIR, test_data, INNER_ONLY=False):
    # Define window names
    win_delaunay = "Delaunay Triangulation"
    cv2.namedWindow(win_delaunay, cv2.WINDOW_NORMAL)
    win_voronoi = "Voronoi Diagram"

    # Turn on animation while drawing triangles
    animate = True

    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)

    # Read in the image.
    if (os.path.exists(os.path.join(ROOT_DIR, test_data))):
        img = cv2.imread(os.path.join(ROOT_DIR, test_data))
    else:
        print('not file founded.')
        exit(0)

    CH = test_data[:-4]
    # Keep a copy around
    img_orig = img.copy()

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an array of points.
    points = []

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)
    h = size[1] - 1
    w = size[0] - 1

    # Read in the points from a text file
    file = np.loadtxt(os.path.join(ROOT_DIR, CH + '_face_open_mouth.txt'))
    file = file.reshape(68, 3)

    for i in range(file.shape[0]):
        if(INNER_ONLY):
            if(i >= 48 and i <= 59): ############## for inner lip only
                continue
        line = file[i]
        x, y, z = line
        points.append((int(float(x)), int(float(y))))


    points.append((0, 0))
    points.append((0, w // 4))
    points.append((0, w // 2))
    points.append((0, w // 4 * 3))
    points.append((0, w))
    points.append((h // 2, w))
    points.append((h, w))
    points.append((h, w // 2))
    points.append((h, 0))
    points.append((h // 4, 0))
    points.append((h // 2, 0))
    points.append((h // 4*3, 0))

    # Insert points into subdiv
    for p in points:
        print(p)
        subdiv.insert(p)

        # Show animation
        if animate:
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay(img_copy, subdiv, (255, 255, 0))
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)

    # Draw delaunay triangles
    draw_delaunay(img, subdiv, (255, 255, 0))
    triangleList = subdiv.getTriangleList()

    p_dict = {}  # Initialize empty dictionary.
    index = 0
    # Draw points
    for p in points:
        # draw_point(img, p, (0, 0, 255))
        p_dict[p] = index
        index = index + 1

    # Allocate space for voronoi Diagram
    img_voronoi = np.zeros(img.shape, dtype=img.dtype)

    # Draw voronoi diagram
    draw_voronoi(img_voronoi, subdiv)

    # Show results
    cv2.imshow(win_delaunay, img)
    print("Press any key to quit...")
    cv2.waitKey(0)

    new_tri = [];

    for line in triangleList:
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        p3 = (line[4], line[5])

        try:
            p1_index = p_dict[p1]
            p2_index = p_dict[p2]
            p3_index = p_dict[p3]
        except:
            continue

        new_tri.append((p1_index, p2_index, p3_index))

    print(new_tri)

    a = np.array(new_tri).astype(int)
    np.savetxt(os.path.join(ROOT_DIR, CH + '_delauney_tri.txt'), a, fmt='%d')








