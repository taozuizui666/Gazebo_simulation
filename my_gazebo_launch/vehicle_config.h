#pragma once

// ================= 配置区：仿真与以后实车 PID 共用 =================
// 仿真直接读取这些宏。PID 应调用 AiCar::wheelTargets(label)，不要再使用 PWM 值作速度。
// 宏的值使用普通数字；修改后重新启动仿真。长度单位 m，速度单位 m/s。
#define AICAR_LABEL_COUNT                  15
#define AICAR_FORWARD_SPEED_MPS            0.2
#define AICAR_MAX_WHEEL_SPEED_RATIO        0.6
#define AICAR_WHEEL_RADIUS_M               0.0325
#define AICAR_WHEEL_SEPARATION_M            0.11
#define AICAR_COMMAND_PUBLISH_HZ           50.0
#define AICAR_COMMAND_TIMEOUT_MS           500
#define AICAR_STOP_COMMAND                 200

// 不模拟电流/PWM。0 表示速度接口不额外限制加速度；转矩是 Gazebo 速度约束的能力。
#define AICAR_SIM_MAX_WHEEL_ACCEL_MPS2      0.0
#define AICAR_SIM_MAX_WHEEL_TORQUE_NM       5.0
#define AICAR_SIM_DRIVE_UPDATE_HZ           100.0
#define AICAR_SIM_TIME_STEP_SEC             0.001

// 模型输入：保持 collect_data / train_model.py 的整数角度分桶和距离截断。
#define AICAR_LIDAR_POINT_COUNT            200
#define AICAR_LIDAR_FRONT_SECTOR_ONLY       1
#define AICAR_LIDAR_MAX_DISTANCE_MM         10000

// RPLIDAR A1M8：顺时针，协议 angle_q6 / 64 度、distance_q2 / 4 毫米。
// MOUNT_YAW 是实物雷达 0 度方向相对车头的逆时针夹角；驱动不能推断安装朝向。
#define AICAR_LIDAR_MOUNT_YAW_DEG           0.0
#define AICAR_LIDAR_X_M                    0.017
#define AICAR_LIDAR_Y_M                    0.0
#define AICAR_LIDAR_Z_M                    0.13
#define AICAR_LIDAR_SCAN_HZ                5.5
#define AICAR_LIDAR_RAW_SAMPLES            1450
#define AICAR_LIDAR_MIN_RANGE_M            0.15
// A1M8 手册/官网的典型测距范围下沿；不是训练数据的硬性过滤阈值。
// 若实测可靠量程不同，只修改此宏对齐仿真传感器。
#define AICAR_LIDAR_PHYSICAL_MAX_RANGE_M    12.0
#define AICAR_LIDAR_NOISE_STDDEV_M          0.01
// 噪声标准差沿用旧模型的 1 cm，尚不是实测误差模型。
// =========================== 配置区结束 ===========================

#include <stdint.h>
#include <math.h>

#define AICAR_STRAIGHT_LABEL (AICAR_LABEL_COUNT / 2)

static_assert(AICAR_LABEL_COUNT >= 3 && AICAR_LABEL_COUNT % 2 == 1 &&
              AICAR_LABEL_COUNT <= AICAR_STOP_COMMAND,
              "Use an odd label count from 3 to STOP_COMMAND.");
static_assert(AICAR_FORWARD_SPEED_MPS > 0, "Forward speed must be positive.");
static_assert(AICAR_MAX_WHEEL_SPEED_RATIO >= 0 &&
              AICAR_MAX_WHEEL_SPEED_RATIO <= 1,
              "Wheel speed ratio must be in [0, 1]; reverse is not used.");
static_assert(AICAR_WHEEL_RADIUS_M > 0 && AICAR_WHEEL_SEPARATION_M > 0,
              "Wheel geometry must be positive.");
static_assert(AICAR_LIDAR_POINT_COUNT >= 4 && AICAR_LIDAR_POINT_COUNT <= 360 &&
              AICAR_LIDAR_POINT_COUNT % 4 == 0,
              "Integer-degree bins require N divisible by 4 and N <= 360.");
static_assert(AICAR_LIDAR_MAX_DISTANCE_MM > 1 &&
              AICAR_LIDAR_MAX_DISTANCE_MM <= 65535, "Distance must fit uint16_t.");

namespace AiCar {
struct WheelTargets {
    double left_mps;
    double right_mps;
    double left_rad_s;
    double right_rad_s;
    double left_rpm;
    double right_rpm;
};

inline bool isSteeringLabel(int label) {
    return label >= 0 && label < AICAR_LABEL_COUNT;
}

// 0 最大左转，中间直行，最后一个最大右转；左右平均轮速恒为 FORWARD_SPEED。
inline WheelTargets wheelTargets(int label) {
    if (!isSteeringLabel(label)) return {};
    const double steering =
        double(AICAR_STRAIGHT_LABEL - label) / AICAR_STRAIGHT_LABEL;
    const double delta =
        AICAR_FORWARD_SPEED_MPS * AICAR_MAX_WHEEL_SPEED_RATIO * steering;
    const double left = AICAR_FORWARD_SPEED_MPS - delta;
    const double right = AICAR_FORWARD_SPEED_MPS + delta;
    const double left_rad_s = left / AICAR_WHEEL_RADIUS_M;
    const double right_rad_s = right / AICAR_WHEEL_RADIUS_M;
    const double rad_s_to_rpm = 60.0 / (2.0 * 3.14159265358979323846);
    return {left, right, left_rad_s, right_rad_s,
            left_rad_s * rad_s_to_rpm, right_rad_s * rad_s_to_rpm};
}

inline bool isActiveLidarIndex(uint16_t index) {
    return index < AICAR_LIDAR_POINT_COUNT &&
        (!AICAR_LIDAR_FRONT_SECTOR_ONLY ||
         index < AICAR_LIDAR_POINT_COUNT / 4 ||
         index > 3 * AICAR_LIDAR_POINT_COUNT / 4);
}

// 原始 RPLIDAR 点：先检查，再丢弃角度小数；与现有 Teensy 分桶相同。
inline int lidarIndex(float angle_degrees) {
    if (!isfinite(angle_degrees) || angle_degrees < 0 || angle_degrees >= 360)
        return -1;
    const uint16_t angle = static_cast<uint16_t>(angle_degrees);
    return static_cast<int>((angle / 360.0f) * AICAR_LIDAR_POINT_COUNT);
}

inline uint16_t lidarDistance(float distance_mm) {
    if (!isfinite(distance_mm) || distance_mm <= 0 ||
        distance_mm >= AICAR_LIDAR_MAX_DISTANCE_MM) return 0;
    return static_cast<uint16_t>(distance_mm);
}
}  // namespace AiCar
