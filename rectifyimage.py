#code=utf-8
# Author: wang liyou
"""
https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/RectifKitE
"""

import os
import cv2
import numpy as np
import scipy.io
from scipy import interpolate


class Rectify:

    def __init__(self, name):
        self._name = name
        self._pml = None
        self._pml = None
        self._points = None
        self._il = None
        self._ir = None

    @staticmethod
    def _load_points(name):
        with open(name, 'r') as f:
            data = f.readlines()
        points = np.array([float(n) for d in data for n in d.split()])
        points = points.reshape(-1, 3)
        return points

    def _load_camera_parameter(self):
        c_path = os.getcwd()
        camera_path = os.path.join(c_path, 'data', "{}_cam.mat".format(self._name))
        points_path = os.path.join(c_path, "data", "{}_points".format(self._name))
        self._pml = scipy.io.loadmat(camera_path)['pml']  # 3 x 4
        self._pmr = scipy.io.loadmat(camera_path)['pmr']
        self._points = self._load_points(points_path)

    def _load_img(self):
        c_path = os.getcwd()
        self._il = cv2.imread(os.path.join(c_path, 'images', "{}0.png".format(self._name)))
        self._ir = cv2.imread(os.path.join(c_path, 'images', "{}1.png".format(self._name)))

    def _draw_line(self, img, c, x1, x2, T=None):
        """
        c[0]*x + c[1]*y + c[2] = 0
        T is a 3x3 matrix encoding a projective transformation of the plane
        :param img:
        :param c:
        :param x1:
        :param x2:
        :param T:
        :return:
        """
        if T is None:
            T = np.eye(3)
        if c[0] == 0:
            c[0] = 1e-12
        if c[1] == 0:
            c[1] = 1e-12

        tmp = np.array([[0, -c[2]/c[0]], [-c[2]/c[1], 0], [1, 1]], dtype=np.float32)
        c3d = T.dot(tmp)  # 3, 2

        x = c3d[0, :]/c3d[2, :]  # 2,
        y = c3d[1, :]/c3d[2, :]

        a = y[1] - y[0]
        b = x[0] - x[1]
        k = -x[0] * a - y[0] * b
        if b == 0:
            b = 1e-12

        y1 = -a / b * x1 - k/b
        y2 = -a / b * x2 - k/b

        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        return img

    def _rectify(self, Po1, Po2, d1=None, d2=None):
        """
        compute rectification matrices in homogeneous coordinate
        [T1,T2,Pn1,Pn2] = rectify(Po1,Po2,d) computes the rectified
         projection matrices "Pn1" and "Pn2", and the transformation
         of the retinal plane "T1" and  "t2" (in homogeneous coord.)
         which perform rectification.  The arguments are the two old
         projection  matrices "Po1" and "Po2" and two 2D displacement
         d1 and d2 wich are applied to the new image centers.
        :param pml:
        :param pmr:
        :param d1:
        :param d2:
        :return:
        """
        if d1 is None:
            d1 = np.zeros((2, ))
        if d2 is None:
            d2 = np.zeros((2, ))
        assert d1[1] == d2[1], "left and right vertical displacements must be the same"
        A1, R1, t1 = self._art(Po1)
        A2, R2, t2 = self._art(Po2)
        c1 = -R1.T.dot(np.linalg.inv(A1).dot(Po1[:, -1]))
        c2 = -R2.T.dot(np.linalg.inv(A2).dot(Po2[:, -1]))
        v1 = c2 - c1
        v2 = np.cross(R1[2, :].T, v1)
        v3 = np.cross(v1, v2)
        R = np.vstack((v1/np.linalg.norm(v1), v2/np.linalg.norm(v2), v3/np.linalg.norm(v3)))
        An1 = A2.copy()
        An1[0, 1] = 0
        An2 = A2.copy()
        An2[0, 1] = 0

        # translate image center
        An1[0, 2] = An1[0, 2] + d1[0]
        An1[1, 2] = An1[1, 2] + d1[1]
        An2[0, 2] = An2[0, 2] + d2[0]
        An2[1, 2] = An2[1, 2] + d2[1]

        Pn1 = An1.dot(np.hstack((R, -R.dot(c1[:, np.newaxis]))))
        Pn2 = An2.dot(np.hstack((R, -R.dot(c2[:, np.newaxis]))))

        T1 = Pn1[0:3, 0:3].dot(np.linalg.inv(Po1[0:3, 0:3]))
        T2 = Pn2[0:3, 0:3].dot(np.linalg.inv(Po2[0:3, 0:3]))

        return T1, T2, Pn1, Pn2

    @staticmethod
    def _art(P, fsign=None):
        """
        ART  Factorize camera matrix into intrinsic and extrinsic matrices
        :param P: 3 * 4 projection matrix P
        :param fsign: as P=A*[R;t] and enforce the sign of the focal length to be fsign
        :return:
        """
        if fsign == None:
            fsign = 1
        s = P[:3, -1]
        Q = np.linalg.inv(P[0:3, 0:3])
        U, B = np.linalg.qr(Q)
        sig = np.sign(B[2, 2])
        B = B * sig
        s = s * sig
        if fsign * B[0, 0] < 0:
            E = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.float32)
            B = E.dot(B)
            U = U.dot(E)

        if fsign * B[1, 1] < 0:
            E = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
            B = E.dot(B)
            U = U.dot(E)

        if np.linalg.det(U) < 0:
            U = -U
            s = -s

        if np.linalg.norm(Q - U.dot(B)) > 1e-10 and np.linalg.norm(Q + U.dot(B)) > 1e-10:
            raise ValueError("Something wrong with the QR factorization.")
        R = U.T
        t = B.dot(s)
        A = np.linalg.inv(B)
        A = A / A[2, 2]
        assert np.linalg.det(R) > 0, "R is not a rotation matrix"
        assert A[2, 2] > 0, "Wrong sign of A(3,3)"
        W = A.dot(np.hstack((R, t[:, np.newaxis])))
        assert np.linalg.matrix_rank(np.hstack((P.T.flatten(), W.T.flatten()))) == 1, "Something wrong with the ART factorization"
        return A, R, t

    @staticmethod
    def _fund(pml, pmr):
        """
        Computes fundamental matrix and epipoles from camera matrices.
        :return:
        """
        cl = -np.linalg.inv(pml[:, :3]).dot(pml[:, -1])
        cr = -np.linalg.inv(pmr[:, :3]).dot(pmr[:, -1])
        cl_homo = np.append(cl, 1)[:, np.newaxis]
        cr_homo = np.append(cr, 1)[:, np.newaxis]
        el = pml.dot(cr_homo)
        er = pmr.dot(cl_homo)
        F = np.array([[0, -er[2], er[1]], [er[2], 0, -er[0]], [-er[1], er[0], 0]], dtype=np.float32).dot(pmr[:, :3].dot(np.linalg.inv(pml[:, :3])))
        F = F/np.linalg.norm(F)
        return F, el, er

    def _mcbb(self, s1, s2, H1, H2):
        """
        MCBB minimum common bounding box
        H1: [3, 3]
        H2: [3, 3]
        bb is the bounding box given as [minx; miny; maxx; maxy]
        s1 is the result of size(I1) [2, ]
        s2 is the result of size(I2) [2, ]
        :param s2:
        :param H1:
        :param H2:
        :return:
        """
        corners = np.array([[0, 0, s1[1], s1[1]], [0, s1[0], 0, s1[0]]], dtype=np.float32)
        corners_x = self._p2t(H1, corners)

        minx = np.floor(np.min(corners_x[0, :]))
        maxx = np.ceil(np.max(corners_x[0, :]))
        miny = np.floor(np.min(corners_x[1, :]))
        maxy = np.ceil(np.max(corners_x[1, :]))
        bb1 = np.array([[minx], [miny], [maxx], [maxy]], dtype=np.float32)

        corners = np.array([[0, 0, s2[1], s2[1]], [0, s2[0], 0, s2[0]]], dtype=np.float32)
        corners_x = self._p2t(H2, corners)
        minx = np.floor(np.min(corners_x[0, :]))
        maxx = np.ceil(np.max(corners_x[0, :]))
        miny = np.floor(np.min(corners_x[1, :]))
        maxy = np.ceil(np.max(corners_x[1, :]))
        bb2 = np.array([[minx], [miny], [maxx], [maxy]], dtype=np.float32)
        q1 = np.min(np.vstack((bb1.T, bb2.T)), axis=0)
        q2 = np.max(np.vstack((bb1.T, bb2.T)), axis=0)
        bb = np.hstack((q1[0:2], q2[2:4]))
        return bb

    @staticmethod
    def _p2t(H, m):
        """
        Applying projection transformation in 2D
        :param H:
        :param m:
        :return:
        """
        na, ma = H.shape
        assert na == 3, 'The format of the transformation matrix is incorrect 3 x 3'
        assert ma == 3, 'The format of the transformation matrix is incorrect 3 x 3'
        rml, cml = m.shape
        assert rml == 2, 'image coordinates must be Cartesian coordinates'
        dime = m.shape[1]
        c3d = np.vstack((m, np.ones((1, dime))))
        h2d = H.dot(c3d)
        c2d = h2d[0:2, :] / np.hstack((h2d[-1, :][:, np.newaxis], h2d[-1, :][:, np.newaxis])).T
        mt = c2d[:2, :]
        return mt

    def _imwarp(self, I, H, meth='binlinear', sz=None):
        """
        I2 = imwarp(I,H) apply the projective transformation specified by H to the image I using linear interpolation
        the output image I2 has the same size of I
        I2 = imwarp(I,H,meth) use  method 'meth' for interpolation
        I2 = imwarp(I,H,meth,sz) yield an output image with specific size. sz can be:
        - 'valid': Make output image I2 large enough to contain the entire rotated image
        - 'same': Make output image I2 the same size as the input image I, cropping the warped image to fit (default).
        - a vector of 4 elements specifying the bounding box
        The output bb is the bounding box of the transformed image in the coordinate frame of the input image
        The first 2 elements of the bb are the translation that have been applied to the upper left corner.
        The bounding box is specified with [minx; miny; maxx; maxy]
        :param I:
        :param H:
        :param meth:
        :param sz:
        :return:
        """
        hm, hn = H.shape
        assert hm == 3, 'Invalid input transformation'
        assert hn == 3, 'Invalid input transformation'

        if sz is None:
            sz = 'same'
        if meth is None:
            meth = 'linear'

        if sz is 'same':
            minx = 0
            maxx = I.shape[1] - 1
            miny = 0
            maxy = I.shape[0] - 1

        if sz is 'valid':
            corners = np.array([[0, 0, I.shape[1], I.shape[1]], [0, I.shape[0], 0, I.shape[0]]])
            corners_x = self._p2t(H, corners)
            minx = np.floor(np.min(corners_x[0, :]))
            maxx = np.ceil(np.max(corners_x[0, :]))
            miny = np.floor(np.min(corners_x[1, :]))
            maxy = np.ceil(np.max(corners_x[1, :]))

        if type(sz) != str:
            minx = sz[0]
            miny = sz[1]
            maxx = sz[2]
            maxy = sz[3]

        bb = np.array([[minx], [miny], [maxx], [maxy]])
        x, y = np.meshgrid(np.linspace(minx, maxx-1, maxx-minx), np.linspace(miny, maxy-1, maxy-miny))
        pp = self._p2t(np.linalg.inv(H), np.vstack((x.T.flatten()[:, np.newaxis].T, y.T.flatten()[:, np.newaxis].T)))
        xi = self._ivec(pp[0, :][:, np.newaxis], x.shape[0])
        yi = self._ivec(pp[1, :][:, np.newaxis], y.shape[0])
        I2 = self._interp2(np.linspace(0, I.shape[1]-1, I.shape[1]), np.linspace(0, I.shape[0]-1, I.shape[0]), I.T, xi, yi)
        alpha = ~np.isnan(I2)
        I2 = I2.astype(np.uint8)

        return I2, bb, alpha

    @staticmethod
    def _interp2(w_in, h_in, img_in, w_out, h_out):
        f = interpolate.RegularGridInterpolator((w_in, h_in), img_in, method='linear',
                                                bounds_error=False, fill_value=np.nan)
        x_flatten = w_out.flatten()
        y_flatten = h_out.flatten()
        xy_new = list(zip(x_flatten, y_flatten))
        img_flatten = f(xy_new)
        img = img_flatten.reshape(w_out.shape)
        return img

    @staticmethod
    def _ivec(v, r):
        """
        Returns matrix A with R rows, Divide the vector V into R parts, each of which Form the column of A. Reverse operation v = a (:)
        If the number of rows of a is known, Number of rows must be Vector size V
        :return:
        """
        n, m = v.shape
        assert m == 1, 'Vector V must be a column vector'
        assert (m * n) % r == 0,  "appropriate number of rows"
        A = v.reshape(r, int(m * n / r), order='F')
        return A

    @staticmethod
    def _skew(x):
        assert len(x) == 3, "Vector must be  3-dimensional"
        X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=np.float32)
        return X

    def run(self):
        self._load_camera_parameter()
        self._load_img()
        F, el, er = self._fund(self._pml, self._pmr)
        TL, TR, pml1, pmr1 = self._rectify(self._pml, self._pmr)
        # centering left image
        p = np.array([self._il.shape[0]/2, self._il.shape[1]/2, 1], dtype=np.float32)
        px = TL.dot(p[:, np.newaxis]).flatten()
        dL = p[0:2] - px[0:2]/px[2]
        # centering right image
        p = np.array([self._ir.shape[0]/2, self._ir.shape[1]/2, 1], dtype=np.float32)
        px = TR.dot(p[:, np.newaxis]).flatten()
        dR = p[0:2] - px[0:2]/px[2]
        # vertical diplacement must be the same
        dL[1] = dR[1]
        TL, TR, pml1, pmr1 = self._rectify(self._pml, self._pmr, dL, dR)
        print("-------------warping----------------------------")
        bb = self._mcbb(self._il.shape, self._ir.shape, TL, TR)
        # warp rgb channel
        JL = np.zeros((int(bb[3] - bb[1]), int(bb[2] - bb[0]), 3))
        JR = np.zeros((int(bb[3] - bb[1]), int(bb[2] - bb[0]), 3))
        for c in range(0, 3):
            img, bbL, alphaL = self._imwarp(self._il[:, :, c], TL, meth='bilinear', sz=bb)
            JL[:, :, c] = img
            img, bbR, alphaR = self._imwarp(self._ir[:, :, c], TR, meth='bilinear', sz=bb)
            JR[:, :, c] = img
        print("-------------warp tie points--------------------")
        mlx = self._p2t(TL, self._points)

        for i in range(0, mlx.shape[1]):
            JL = cv2.circle(JL, (int(mlx[0, i] - bbL[0]), int(mlx[1, i] - bbL[1])), color=(0, 255, 255),
                            radius=2, thickness=2)

        for i in range(0, mlx.shape[1]):
            liner = self._skew([1, 0, 0]).dot(np.append(mlx[:, i] - bbL[0:2].flatten(), 1)[:, np.newaxis])
            JR = self._draw_line(JR, liner, 0, JR.shape[1], None)
        J_combined = np.hstack((JL.astype(np.uint8), JR.astype(np.uint8)))
        cv2.imwrite(os.path.join(os.getcwd(), "imagesE", "python_{}_0.jpg".format(self._name)), JL)
        cv2.imwrite(os.path.join(os.getcwd(), "imagesE", "python_{}_1.jpg".format(self._name)), JR)
        cv2.namedWindow("Rectify Image")
        cv2.imshow("Rectify Image", J_combined)
        cv2.waitKey(0)


if __name__ == "__main__":
    name = 'Sport'
    rectify = Rectify(name)
    rectify.run()