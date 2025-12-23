"""
极简3D引擎 - 教学版
用于演示线性代数在3D图形中的应用

核心概念：
1. 向量和矩阵运算
2. 坐标变换（模型-视图-投影）
3. 光照计算
4. 3D到2D的投影
"""

import numpy as np
import pygame
from pygame.locals import *


# ============ 第一部分：数学基础 - 向量和矩阵运算 ============

class Vector3:
    """3维向量类 - 用于表示点、方向、颜色等"""
    
    def __init__(self, x=0, y=0, z=0):
        self.data = np.array([x, y, z], dtype=float)
    
    @property
    def x(self):
        return self.data[0]
    
    @x.setter
    def x(self, value):
        self.data[0] = value
    
    @property
    def y(self):
        return self.data[1]
    
    @y.setter
    def y(self, value):
        self.data[1] = value
    
    @property
    def z(self):
        return self.data[2]
    
    @z.setter
    def z(self, value):
        self.data[2] = value
    
    def length(self):
        """向量长度（模）"""
        return np.sqrt(np.sum(self.data ** 2))
    
    def normalize(self):
        """归一化向量 - 返回单位向量"""
        length = self.length()
        if length > 0:
            return Vector3(*(self.data / length))
        return Vector3()
    
    def dot(self, other):
        """点积 - 用于计算投影和角度"""
        return np.dot(self.data, other.data)
    
    def cross(self, other):
        """叉积 - 用于计算垂直向量"""
        result = np.cross(self.data, other.data)
        return Vector3(*result)
    
    def __add__(self, other):
        return Vector3(*(self.data + other.data))
    
    def __sub__(self, other):
        return Vector3(*(self.data - other.data))
    
    def __mul__(self, scalar):
        return Vector3(*(self.data * scalar))
    
    def __repr__(self):
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


def create_rotation_matrix(angle_x, angle_y, angle_z):
    """
    创建旋转矩阵 - 欧拉角旋转
    
    参数:
        angle_x, angle_y, angle_z: 绕x、y、z轴的旋转角度（弧度）
    
    返回:
        4x4旋转矩阵
    """
    # 绕X轴旋转
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x), 0],
        [0, np.sin(angle_x), np.cos(angle_x), 0],
        [0, 0, 0, 1]
    ])
    
    # 绕Y轴旋转
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y), 0],
        [0, 1, 0, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y), 0],
        [0, 0, 0, 1]
    ])
    
    # 绕Z轴旋转
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0, 0],
        [np.sin(angle_z), np.cos(angle_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # 组合旋转：先Z，再Y，最后X
    return Rx @ Ry @ Rz


def create_translation_matrix(x, y, z):
    """创建平移矩阵"""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def create_scale_matrix(sx, sy, sz):
    """创建缩放矩阵"""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


# ============ 第二部分：摄像机系统 ============

class Camera:
    """
    摄像机类 - 实现视图变换和投影变换
    
    视图变换：将世界坐标转换到摄像机坐标
    投影变换：将3D坐标投影到2D屏幕
    """
    
    def __init__(self, position, look_at, up, fov=60, aspect=1.333, near=0.1, far=100):
        """
        参数:
            position: 摄像机位置（Vector3）
            look_at: 摄像机看向的点（Vector3）
            up: 向上方向（Vector3）
            fov: 视场角（度）
            aspect: 宽高比
            near: 近裁剪面
            far: 远裁剪面
        """
        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
    
    def get_view_matrix(self):
        """
        计算视图矩阵（View Matrix）
        将世界坐标系转换到摄像机坐标系
        """
        # 计算摄像机的三个坐标轴
        # forward: 摄像机看向的方向（-Z轴）
        forward = (self.position - self.look_at).normalize()
        # right: 摄像机的右方向（X轴）
        right = self.up.cross(forward).normalize()
        # up: 摄像机的真实上方向（Y轴）
        up = forward.cross(right)
        
        # 构建视图矩阵 = 旋转矩阵 * 平移矩阵
        view_matrix = np.array([
            [right.x, right.y, right.z, -right.dot(self.position)],
            [up.x, up.y, up.z, -up.dot(self.position)],
            [forward.x, forward.y, forward.z, -forward.dot(self.position)],
            [0, 0, 0, 1]
        ])
        
        return view_matrix
    
    def get_projection_matrix(self):
        """
        计算透视投影矩阵（Perspective Projection Matrix）
        将3D坐标投影到2D屏幕，实现近大远小的透视效果
        """
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        projection_matrix = np.array([
            [f / self.aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far + self.near) / (self.near - self.far), 
             (2 * self.far * self.near) / (self.near - self.far)],
            [0, 0, -1, 0]
        ])
        
        return projection_matrix


# ============ 第三部分：光照系统 ============

class Light:
    """
    定向光源 - 实现简单的Phong光照模型
    
    Phong模型包括三个分量：
    1. 环境光（Ambient）：模拟间接光照
    2. 漫反射（Diffuse）：模拟物体表面的漫反射
    3. 镜面反射（Specular）：模拟高光
    """
    
    def __init__(self, direction, color, intensity=1.0):
        """
        参数:
            direction: 光照方向（Vector3）
            color: 光照颜色（RGB，0-1）
            intensity: 光照强度
        """
        self.direction = direction.normalize()
        self.color = np.array(color)
        self.intensity = intensity
    
    def calculate_lighting(self, point, normal, view_dir, material_color):
        """
        计算点的光照强度
        
        参数:
            point: 被照射点的位置
            normal: 该点的法向量
            view_dir: 观察方向
            material_color: 材质颜色（RGB，0-1）
        
        返回:
            最终颜色（RGB，0-255）
        """
        # 1. 环境光
        ambient = 0.2 * material_color
        
        # 2. 漫反射（Diffuse） - Lambert余弦定律
        # 光照强度与法向量和光线方向的夹角的余弦成正比
        light_dir = self.direction * -1  # 光线指向光源
        diffuse_intensity = max(0, normal.dot(light_dir))
        diffuse = diffuse_intensity * material_color * self.color * self.intensity
        
        # 3. 镜面反射（Specular） - Phong反射模型
        # 反射向量：R = 2(N·L)N - L
        reflect_dir = normal * (2 * normal.dot(light_dir)) - light_dir
        specular_intensity = max(0, reflect_dir.dot(view_dir)) ** 32  # 32是高光指数
        specular = specular_intensity * self.color * self.intensity * 0.5
        
        # 组合三个分量
        final_color = ambient + diffuse + specular
        
        # 限制在0-1范围内，然后转换到0-255
        final_color = np.clip(final_color, 0, 1) * 255
        
        return tuple(final_color.astype(int))


# ============ 第四部分：3D物体 - 立方体 ============

class Box:
    """
    立方体类 - 用8个顶点和12个三角形面表示
    
    演示了3D物体如何存储和渲染
    """
    
    def __init__(self, center, size, color=(100, 150, 200)):
        """
        参数:
            center: 立方体中心位置（Vector3）
            size: 立方体边长
            color: 材质颜色（RGB，0-255）
        """
        self.center = center
        self.size = size
        self.color = np.array(color) / 255.0  # 转换到0-1范围
        
        # 定义立方体的8个顶点（相对于中心）
        s = size / 2
        self.vertices = [
            Vector3(-s, -s, -s),  # 0: 左下后
            Vector3(s, -s, -s),   # 1: 右下后
            Vector3(s, s, -s),    # 2: 右上后
            Vector3(-s, s, -s),   # 3: 左上后
            Vector3(-s, -s, s),   # 4: 左下前
            Vector3(s, -s, s),    # 5: 右下前
            Vector3(s, s, s),     # 6: 右上前
            Vector3(-s, s, s),    # 7: 左上前
        ]
        
        # 定义12个三角形面（每个面2个三角形，顶点顺序保证法线朝外）
        self.faces = [
            # 后面 (Z-)，法线朝-Z
            (0, 2, 1), (0, 3, 2),
            # 前面 (Z+)，法线朝+Z
            (4, 5, 6), (4, 6, 7),
            # 左面 (X-)，法线朝-X
            (0, 7, 3), (0, 4, 7),
            # 右面 (X+)，法线朝+X
            (1, 2, 6), (1, 6, 5),
            # 底面 (Y-)，法线朝-Y
            (0, 1, 5), (0, 5, 4),
            # 顶面 (Y+)，法线朝+Y
            (3, 6, 2), (3, 7, 6),
        ]
        
        self.rotation = Vector3(0, 0, 0)  # 旋转角度（弧度）
    
    def get_face_normal(self, face_indices):
        """
        计算三角形面的法向量
        使用叉积计算垂直于平面的向量
        """
        v0 = self.vertices[face_indices[0]]
        v1 = self.vertices[face_indices[1]]
        v2 = self.vertices[face_indices[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = edge1.cross(edge2).normalize()
        
        return normal
    
    def get_world_vertices(self):
        """
        获取世界坐标系中的顶点位置
        应用旋转和平移变换
        """
        # 创建变换矩阵
        rotation_matrix = create_rotation_matrix(
            self.rotation.x, self.rotation.y, self.rotation.z
        )
        translation_matrix = create_translation_matrix(
            self.center.x, self.center.y, self.center.z
        )
        
        # 组合变换：先旋转，后平移
        transform = translation_matrix @ rotation_matrix
        
        # 应用变换到每个顶点
        world_vertices = []
        for v in self.vertices:
            # 将Vector3转换为齐次坐标（添加w=1）
            v_homogeneous = np.array([v.x, v.y, v.z, 1])
            # 应用变换矩阵
            v_transformed = transform @ v_homogeneous
            # 转换回Vector3
            world_vertices.append(Vector3(*v_transformed[:3]))
        
        return world_vertices


# ============ 第五部分：渲染引擎 ============

class Engine3D:
    """
    3D渲染引擎主类
    整合摄像机、光照和物体，实现完整的渲染流程
    """
    
    def __init__(self, width=800, height=600):
        """初始化引擎和Pygame窗口"""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("极简3D引擎 - 教学版")
        
        # 创建摄像机（位置、看向点、向上方向）
        self.camera = Camera(
            position=Vector3(0, 2, 5),
            look_at=Vector3(0, 0, 0),
            up=Vector3(0, 1, 0),
            fov=60,
            aspect=width / height
        )
        
        # 创建光源（方向、颜色、强度）
        self.light = Light(
            direction=Vector3(1, -1, -1),
            color=(1.0, 1.0, 0.9),  # 淡黄色光
            intensity=0.8
        )
        
        # 创建立方体
        self.box = Box(
            center=Vector3(0, 0, 0),
            size=1.5,
            color=(100, 150, 200)
        )
        
        # 鼠标拖动状态
        self.mouse_down = False
        self.last_mouse_pos = None
        
        self.clock = pygame.time.Clock()
    
    def project_to_screen(self, point):
        """
        将3D世界坐标投影到2D屏幕坐标
        
        完整的变换流程：
        世界坐标 -> 视图坐标 -> 裁剪坐标 -> NDC坐标 -> 屏幕坐标
        """
        # 转换为齐次坐标
        point_homogeneous = np.array([point.x, point.y, point.z, 1])
        
        # 1. 应用视图变换
        view_matrix = self.camera.get_view_matrix()
        view_space = view_matrix @ point_homogeneous
        
        # 2. 应用投影变换
        projection_matrix = self.camera.get_projection_matrix()
        clip_space = projection_matrix @ view_space
        
        # 3. 透视除法：转换到NDC坐标（-1到1）
        if clip_space[3] != 0:
            ndc = clip_space[:3] / clip_space[3]
        else:
            return None
        
        # 4. 转换到屏幕坐标（0到width/height）
        screen_x = (ndc[0] + 1) * 0.5 * self.width
        screen_y = (1 - ndc[1]) * 0.5 * self.height  # Y轴翻转
        
        # 返回屏幕坐标和深度值
        return (int(screen_x), int(screen_y), ndc[2])
    
    def render_box(self):
        """渲染立方体"""
        # 获取世界坐标系中的顶点
        world_vertices = self.box.get_world_vertices()
        
        # 投影所有顶点到屏幕
        projected_vertices = []
        for v in world_vertices:
            proj = self.project_to_screen(v)
            if proj is None:
                return  # 跳过无效投影
            projected_vertices.append(proj)
        
        # 准备渲染的面（按深度排序）
        faces_to_draw = []
        
        for face in self.box.faces:
            # 使用世界坐标的顶点计算法向量
            v0 = world_vertices[face[0]]
            v1 = world_vertices[face[1]]
            v2 = world_vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = edge1.cross(edge2).normalize()
            
            # 背面剔除：只渲染面向摄像机的面
            face_center = (v0 + v1 + v2) * (1/3)
            view_dir = (self.camera.position - face_center).normalize()
            
            if normal.dot(view_dir) > 0:  # 面向摄像机
                # 计算光照
                color = self.light.calculate_lighting(
                    face_center, normal, view_dir, self.box.color
                )
                
                # 计算平均深度（用于排序）
                avg_depth = sum(projected_vertices[i][2] for i in face) / 3
                
                # 获取屏幕坐标
                points = [(projected_vertices[i][0], projected_vertices[i][1]) 
                         for i in face]
                
                faces_to_draw.append((avg_depth, points, color))
        
        # 从远到近绘制面（画家算法）
        faces_to_draw.sort(key=lambda x: x[0], reverse=True)
        
        for _, points, color in faces_to_draw:
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (0, 0, 0), points, 1)  # 绘制边框
    
    def draw_text_info(self):
        """显示信息文本"""
        # 使用系统中文字体
        try:
            font = pygame.font.SysFont('microsoftyahei', 20)  # 微软雅黑
        except:
            try:
                font = pygame.font.SysFont('simhei', 20)  # 黑体
            except:
                font = pygame.font.Font(None, 24)  # 后备方案
        
        info_texts = [
            "极简3D引擎 - 线性代数演示",
            f"旋转: X={np.degrees(self.box.rotation.x):.0f}° "
            f"Y={np.degrees(self.box.rotation.y):.0f}° "
            f"Z={np.degrees(self.box.rotation.z):.0f}°",
            "鼠标拖动: 旋转立方体   ESC: 退出"
        ]
        
        for i, text in enumerate(info_texts):
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i * 25))
    
    def run(self):
        """主循环"""
        running = True
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键
                        self.mouse_down = True
                        self.last_mouse_pos = event.pos
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_down = False
                        self.last_mouse_pos = None
                elif event.type == MOUSEMOTION:
                    if self.mouse_down and self.last_mouse_pos:
                        # 计算鼠标移动距离
                        dx = event.pos[0] - self.last_mouse_pos[0]
                        dy = event.pos[1] - self.last_mouse_pos[1]
                        
                        # 根据移动距离更新旋转角度
                        # 水平移动控制Y轴旋转，垂直移动控制X轴旋转
                        self.box.rotation.y += dx * 0.01
                        self.box.rotation.x += dy * 0.01
                        
                        self.last_mouse_pos = event.pos
            
            # 渲染
            self.screen.fill((20, 20, 40))  # 深蓝色背景
            self.render_box()
            self.draw_text_info()
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()


# ============ 主程序入口 ============

if __name__ == "__main__":
    print("=" * 60)
    print("极简3D引擎启动")
    print("=" * 60)
    print("\n这个程序演示了以下线性代数概念：")
    print("1. 向量运算：加法、减法、点积、叉积")
    print("2. 矩阵变换：旋转、平移、缩放")
    print("3. 坐标系变换：模型空间 -> 世界空间 -> 视图空间 -> 裁剪空间")
    print("4. 光照计算：法向量、反射、点积应用")
    print("5. 投影：3D到2D的透视投影\n")
    print("=" * 60)
    
    # 创建并运行引擎
    engine = Engine3D(width=800, height=600)
    engine.run()
