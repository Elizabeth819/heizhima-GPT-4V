import os
import requests
import base64
import json
import cv2
 
# Configuration
GPT4V_KEY = "8a4b74039ec344fea9505eaccff23f1f"
GPT4V_ENDPOINT = "https://gpt4vliz-switnor-vm.openai.azure.com/openai/deployments/gpt-4v/extensions/chat/completions?api-version=2023-12-01-preview"
IMAGE_PATH = "./img_scenes/00003.jpg"

# convert sample image to base64 string for system prompt (i.e. FSL)
sample_image_3 = cv2.imread('./img_scenes/00003.jpg')
-, buffer = cv2.imencode(' -jpg', sample_image)
base64_image_3 = base64.b64encode(buffer).decode('utf-8')
sample_image_4 = cv2.imread('./img_scenes/00004.jpg')
-, buffer = cv2.imencode(' -jpg', sample_image)
base64_image_4 = base64.b64encode(buffer).decode('utf-8')

system_prompt = '''
                You are a sophisticated AI designd to assist autonomous driving systems by interpreting visual data from road scenes. 
                Your primary task is to identify the descriptive hashtags that correlate to specific elements and conditions visible in the image. 
                These hashtags will inform the autonomous system of the driving environment, ensuring better decision-making for safer navigation.
                Always produce your answer in JSON format.
                
                The following is a comprehensive list of labels (keys) and their corresponding scene values (Chinese contents) you should use:
                    标签中文内容 - 场景值 key - 场景值英文内容
                    停车场类型 - 地上 - ground
                    停车场类型 - 地下 - underground
                    停车场类型 - 停车楼 - parking_lot
                    停车位类型 - 垂直车位 - vertical_parking_space
                    停车位类型 - 水平车位 - horizontal_parking_space
                    停车位类型 - 斜向车位 - inclined_parking_space
                    停车位类型 - 特殊车位 - special_parking_space
                    停车位类型 - T字车位 - T-shaped_parking_space
                    停车位类型 - 地锁车位 - Ground lock parking space
                    划线颜色 - 白色 - white
                    划线颜色 - 黄色 - yellow
                    划线颜色 - 红色 - red
                    划线颜色 - 绿色 - green
                    划线颜色 - 黑色 - black
                    地面类型 - 水泥地面 - concrete_floor
                    地面类型 - 沥青地面 - asphalt_ground
                    地面类型 - 砖块草地 - brick_grass
                    地面类型 - 环氧地坪 - epoxy_floor
                    道路区域 - 高速 - high_speed
                    道路区域 - 城区 - city_proper
                    道路区域 - 乡间 - country
                    道路区域 - 山路 - mountain_road
                    道路区域 - 匝道 - ramp_road
                    天气 - 晴天 - sunny
                    天气 - 小雨 - light_rain
                    天气 - 阴天 - overcast
                    天气 - 小雪 - light_snow
                    天气 - 小雾 - light_foggy
                    天气 - 沙尘暴 - sandstorm
                    天气 - 雾霾 - smog
                    光照 - 弱光 - weak_light
                    光照 - 强光 - strong_light
                    光照 - 顺光 - follow_light
                    光照 - 侧光 - lateral_light
                    光照 - 逆光 - back_light
                    时间 - 黎明 - dawn
                    时间 - 白天 - day
                    时间 - 傍晚 - Evening
                    时间 - 夜晚 - night
                    车道线类型 - 无车道线 - laneway_line
                    车道线类型 - 白色实线 - solid_white_line
                    车道线类型 - 单黄实线 - Single_Yellow_Line
                    车道线类型 - 双黄实线 - Double_Yellow_Line
                    车道线类型 - 黄色虚实线 - Yellow_dotted_line
                    车道线类型 - 黄色禁止停车实线 - Yellow_No_Stopping_Solid_Line
                    车道线类型 - 导流线 - Diversion_line
                    车道线类型 - 禁停网格线 - Stop-grid
                    车道线类型 - 白色虚线 - dotted_white_line
                    车道线类型 - 人行横道 - crosswalk
                    车道线类型 - 车道隔离带 - lane_divider
                    车道线类型 - 鱼骨线 - Yugu line
                    车道线类型 - 双黄虚线 - Double dotted line
                    车道线类型 - 单黄虚线 - Yellow dotted line
                    道路角度 - 直道 - Straights
                    道路角度 - 弯道 - winding_road
                    道路角度 - 高速上下口 - High-speed_upper_and_lower_ports
                    道路角度 - 上下坡 - Up-and-down_slope
                    交叉口 - 三岔口 - Sancha
                    交叉口 - 十字路 - Crosscut
                    交叉口 - 多叉口 - Multiple_forks
                    交叉口 - 丁字路口 - T-junction
                    交叉口 - 环岛 - roundabout
                    环境复杂度 - 路两边建筑物 - Buildings_on_both_sides_of_the_road
                    环境复杂度 - 路两边绿植 - Green_plants_on_both_sides_of_the_road
                    环境复杂度 - 路两边广告牌 - Billboards_on_both_sides_of_the_road
                    环境复杂度 - 特殊装饰 - Special_decoration
                    环境复杂度 - 反光的玻璃幕墙 - Reflective_glass_curtain_wall
                    物体类型 - 路障 - roadblock
                    物体类型 - 施工区域 - Construction_area
                    物体类型 - 箭头交通灯 - Arrow_traffic_lights
                    物体类型 - 特殊交通标识 - Special_traffic_sign
                    道路类型 - 高架 - overhead
                    道路类型 - 隧道 - tunnel
                    道路类型 - 桥梁 - bridge
                    道路类型 - 涵洞 - culvert
                    道路类型 - 公交车道 - Bus_lane
                    道路类型 - 专用道 - Dedicated_lane
                    道路类型 - 应急车道 - Emergency_Vehicle_Lane
                    道路类型 - 匝道 - freeway_ramp
                    道路类型 - 多乘员车道 - Multi-occupant_lane
                    道路类型 - 单行道 - One_way_traffic
                    道路类型 - 限行道路 - Restricted_roads
                    道路类型 - 非机动车道 - Non-motorized_lane
                    道路类型 - 潮汐车道 - reversible lanes
                    道路类型 - 柏油路 - tarred_road
                    路面 - 水泥路 - Cement_road
                    路面 - 砂石路 - Gravel_road
                    路面 - 泥巴路 - Mud_Road
                    路面 - 干燥 - dry
                    路面 - 积雪 - snow
                    路面 - 潮湿 - damp
                    路面 - 坑洼 - pothole
                    路面 - 阴影 - shadow
                    路面 - 积水反光 - Ponding_reflection
                    路面 - 路面数字 - Pavement number
                    路面 - 路面文字 - Pavement characters
                    路面 - 积水 - Water
                    道路面 - 城镇 - urban
                    道路面 - 郊区 - suburban
                    道路面 - 乡村 - rural
                    道路面 - 高架 - elevated_road
                    白天/夜晚 - 白天 - Day
                    白天/夜晚 - 夜晚 - Night
                    比例正确/错误 - 正确 - Right
                    比例正确/错误 - 错误 - Wrong
                    帽子颜色 - 黑色 - Black
                    帽子颜色 - 白色 - White
                    帽子颜色 - 花色 - Motley
                    是否有帽沿 - 有 - Have
                    是否有帽沿 - 无 - No
                    眼镜反光 - 低反光 - Low
                    眼镜反光 - 中反光 - Middle
                    眼镜反光 - 高反光 - High
                    口罩颜色 - 黑色 - Black
                    口罩颜色 - 白色 - White
                    口罩颜色 - 花色 - Motley
                    场景 - 商业办公区 - BusinessOffice
                    场景 - 住宅 - Residence
                    场景 - 购物中心 - Mall
                    场景 - 电影院 - MovieTheater
                    场景 - 超市 - Supermarket
                    场景 - 机场 - Airport
                    道路类型 - 窄路 - NarrowRoad
                    道路类型 - 宽阔道路 - WideRoad
                    照明类型 - 明亮照明 - BrightLighting
                    照明类型 - 暗淡照明 - DimLighting
                    照明类型 - 运动传感器照明 - MotionSensorLighting
                    地面类型 - 反光地面 - ReflectiveGround
                    地面类型 - 非反光地面 - Non-reflectiveGround



                ## Example
                Here are two examples photo encoded in Base64 string
                 {base64_image_3}
                 {base64_image_4}
                Here's the output for the above photo:
                [
                    {"id":1, "划线颜色": "白色", "地面类型": "水泥地面", "道路区域": "城区", "天气": "晴朗", "光照": "弱光", "时间": "夜晚", "车道线类型": "白色实线", "道路角度": "直道", "道路灯": "有路灯", "环境复杂度": "路两边绿植", "路面": "水泥路", "道路面": "城镇", "地面类型_2": "非反光地面", "障碍物": "不包含细长障碍物", "有无停车动作": "无停车动作"}, 
                    ("id":2, "划线颜色": "白色", "地面类型": "水泥地面", "道路区域": "城区", "天气": "阴天", "光照": "弱光", "时间": "白天", "车道线类型": "白色实线", "交叉口": "三岔口", "道路灯": "无路灯", "环境复杂度": "路两边绿植", "路面": "水泥路", "道路面": "城镇", "地面类型": "反光地面", "障碍物": "包含细长障碍物", "有无停车动作": "无停车动作"}
                ]                

                '''
encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}
 
# Payload for the request
payload = {
    "enhancements": {
        "ocr": {
            "enabled": True
        },
        "grounding": {
            "enabled": True
        }
    },
    # "dataSources": [
    # {
    #     "type": "AzureComputerVisionVideoIndex",
    #     "parameters": {
    #         "endpoint": "<your_computer_vision_endpoint>",
    #         "key": "<your_computer_vision_key>",
    #         "computerVisionBaseUrl": "<your_computer_vision_endpoint>",
    #         "computerVisionApiKey": "<your_computer_vision_key>",
    #         "indexName": "<name_of_your_index>",
    #         "videoUrls": ["<your_video_SAS_URL>"]
    # }}],
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                #    "text": "As an AI assistant, provide a clear, detailed sentence describing the content depicted in this image. Extract receipts as json: total, currencyCode, phoneNumber."
                    "text": "你是一个气象专家，描述图片中的气象现象，别胡编乱造，只讲事实，用中文"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                },
                {
                    "type": "text",
                    "text": """Describe the image"""
                    # "text": "Extract receipts as json: total, currencyCode, phoneNumber."
                }
            ]
        },
    ],
    "temperature": 0, #发散性。较高的值（例如 0.7）将使输出更随机，并产生更多发散的响应，而较小的值（例如 0.2）将使输出更加集中和具体。而要生成法律文件的话，建议使用低得多的温度。
    #temp: 图片设为0，就描述图片，每次不变。
    "top_p": 0, #控制模型响应的随机性，但它的控制方式有所不同。每次选取可能性大于x%的内容进入输出范围。 一般建议一次只更改这两个参数其中之一，而不是同时更改它们。
    "max_tokens": 2000
}
 
 
# Send request
try:
    response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
    # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    response.raise_for_status()
except requests.RequestException as e:
    raise SystemExit(f"Failed to make the request. Error: {e}")
 
# Handle the response as needed (e.g., print or process)
print(IMAGE_PATH)

if response.status_code == 200:
    # 直接从响应中提取 JSON 数据
    json_data = response.json()
    # 现在 json_data 是一个 Python 字典，您可以按照需要处理它
    formatted_data=json.dumps(json_data, indent=4)
    print(formatted_data)

else:
    print(f"请求失败，状态码：{response.status_code}")    

# print(response.json())