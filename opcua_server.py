import logging
import time
import sys

sys.path.insert(0, "..")
from opcua import ua, Server

# -------------------------------------------------------------------
# 1. 配置服务器
# -------------------------------------------------------------------

# 服务器监听的地址和端口
# 如果你的检测系统和服务器在同一台电脑上，用'0.0.0.0'或'127.0.0.1'都可以
# '0.0.0.0'允许网络上其他电脑访问，'127.0.0.1'只允许本机访问
# SERVER_ENDPOINT = "opc.tcp://172.30.32.231:4840/SafetyServer/server/"
SERVER_ENDPOINT = "opc.tcp://0.0.0.0:4840/SafetyServer/server/"

# 服务器的名称，会显示在客户端
SERVER_NAME = "Worker Safety Detection System Server"

# 命名空间URI，像一个唯一的域名，用于组织你自己的变量
NAMESPACE_URI = "http://solutions.com/workersafety-server"

def create_server():
    """创建一个配置好的OPC UA服务器实例"""
    # 设置日志，只显示警告及以上信息，避免刷屏
    logging.basicConfig(level=logging.WARN)

    # 创建服务器实例
    server = Server()
    server.set_endpoint(SERVER_ENDPOINT)
    server.set_server_name(SERVER_NAME)

    # 设置安全策略 (NoSecurity最简单，适用于安全的内部网络)
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])

    # 注册我们自己的命名空间，并获取它的索引(idx)
    idx = server.register_namespace(NAMESPACE_URI)
    
    print(f"OPC UA Server '{SERVER_NAME}' configured at {SERVER_ENDPOINT}")
    print(f"Namespace '{NAMESPACE_URI}' registered with index {idx}")
    
    return server, idx


# -------------------------------------------------------------------
# 2. 创建我们需要的变量 (Tags)
# -------------------------------------------------------------------

def create_address_space(server, idx):
    """在服务器上创建我们的文件夹和变量"""
    
    # 在 'Objects' 节点下创建一个顶层文件夹来存放我们所有的数据
    main_folder = server.nodes.objects.add_folder(idx, "SafetySystem")

    # 为不同的摄像头或区域创建子文件夹，这样结构更清晰
    cam_default_folder = main_folder.add_folder(idx, "DefaultCamera")
    cam_58_folder = main_folder.add_folder(idx, "Camera_58")
    # 你可以根据需要添加更多摄像头的文件夹
    # cam_125_folder = main_folder.add_folder(idx, "Camera_125")

    print("Creating variables in the address space...")

    # --- 在默认摄像头文件夹下创建变量 ---
    # 我们用一个变量来存储最新的警报代码
    # 初始值为0，表示没有警报
    # 1: 摔倒, 2: 未戴安全帽, 3: 静止, 4: 吸烟
    default_alert_code = cam_default_folder.add_variable(idx, "LastAlertCode", 0)
    
    # 也可以为每种警报类型创建单独的布尔型变量（0或1）
    default_fall_alert = cam_default_folder.add_variable(idx, "FallAlert", False)
    default_no_helmet_alert = cam_default_folder.add_variable(idx, "NoHelmetAlert", False)
    
    # 让这些变量可以被客户端写入
    default_alert_code.set_writable()
    default_fall_alert.set_writable()
    default_no_helmet_alert.set_writable()

    # --- 在特定摄像头(例如Cam_58)文件夹下创建变量 ---
    cam58_alert_code = cam_58_folder.add_variable(idx, "LastAlertCode", 0)
    cam58_fall_alert = cam_58_folder.add_variable(idx, "FallAlert", False)
    cam58_no_helmet_alert = cam_58_folder.add_variable(idx, "NoHelmetAlert", False)
    
    cam58_alert_code.set_writable()
    cam58_fall_alert.set_writable()
    cam58_no_helmet_alert.set_writable()

    print("Address space created successfully.")


# -------------------------------------------------------------------
# 3. 启动服务器并保持运行
# -------------------------------------------------------------------

if __name__ == "__main__":
    # 步骤 1: 创建和配置服务器
    my_server, namespace_index = create_server()

    # 步骤 2: 在服务器上创建地址空间（文件夹和变量）
    create_address_space(my_server, namespace_index)

    # 步骤 3: 启动服务器
    my_server.start()
    
    print("\nOPC UA Server is running. Press Ctrl+C to stop.")

    try:
        # 让服务器一直运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping OPC UA Server...")
    finally:
        # 关闭服务器
        my_server.stop()
        print("Server stopped.")