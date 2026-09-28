/**
 * @file    sg90_driver.c
 * @brief   SG90 舵机驱动 (基于 HAL 定时器 PWM 输出)
 * @note    适用于 STM32H750VBT6，支持通用定时器 PWM 通道
 *
 *          SG90 舵机控制原理：
 *          - PWM 周期：20ms (50Hz)
 *          - 脉宽 0.5ms → 0°，脉宽 2.5ms → 180°
 *          - 脉宽与角度呈线性关系：pulse_us = 500 + angle * (2000 / 180)
 *
 *          校准建议：
 *          - 实际舵机存在个体差异，脉宽范围可能需要微调
 *          - 建议先使用 SG90_SetPulseMicroseconds() 手动确认 0° 和 180° 的精确脉宽
 *          - 电源电压 4.8V~6V 时性能最佳，避免从 MCU 3.3V 直接供电
 */

#include "sg90_driver.h"

/************************ 内部辅助函数 ************************/

/**
 * @brief  计算单个 PWM 计数周期的时间步长 (纳秒)
 * @param  hsg: 舵机句柄指针
 * @retval 每个计数值对应的时间 (ns)
 * @note   step_ns = 1000000000 / (timer_freq / (prescaler + 1))
 */
static uint32_t SG90_GetStepNS(const SG90_HandleTypeDef *hsg)
{
    /* 防止 prescaler 为 0 时的除零 */
    uint32_t psc = (hsg->prescaler > 0U) ? hsg->prescaler : 1U;
    /* timer_freq / psc = 定时器计数频率 (Hz) */
    /* 1e9 / count_freq = 每个计数的纳秒数 */
    uint32_t count_freq = hsg->timer_freq / psc;
    if (count_freq == 0U) return 0U;
    return 1000000000U / count_freq;
}

/************************ 初始化函数 ************************/

/**
 * @brief  STM32H750专用：SG90舵机初始化 (需在CubeMX中配置好定时器PWM通道)
 * @param  hsg: 舵机句柄指针
 * @param  htim: 定时器句柄指针 (需已在CubeMX中配置PWM输出)
 * @param  channel: 定时器通道 (TIM_CHANNEL_1 / TIM_CHANNEL_2 / TIM_CHANNEL_3 / TIM_CHANNEL_4)
 * @param  timer_freq: 定时器输入时钟频率 (Hz)，STM32H750@480MHz时，TIM1/TIM2/TIM3/TIM4/TIM5/TIM8 通常为 240000000
 * @param  prescaler: CubeMX中配置的预分频值
 * @param  period: CubeMX中配置的自动重装载值 (ARR)
 * @retval HAL_OK(0) - 成功, 其他值 - 失败
 * @note   1. 调用前需确保 CubeMX 已配置对应定时器的 PWM 输出模式
 *         2. timer_freq = APBx_Timer_Clock，H750@480MHz 时 APB1/APB2 Timer 时钟为 240MHz
 *         3. 初始化后舵机自动转到 90° (中立位)
 */
HAL_StatusTypeDef SG90_Init(SG90_HandleTypeDef *hsg,
                            TIM_HandleTypeDef *htim,
                            uint32_t channel,
                            uint32_t timer_freq,
                            uint32_t prescaler,
                            uint32_t period)
{
    if (hsg == (void *)0 || htim == (void *)0) {
        return HAL_ERROR;
    }
    if (timer_freq == 0U || period == 0U) {
        return HAL_ERROR;
    }

    /* 保存配置 */
    hsg->htim        = htim;
    hsg->channel     = channel;
    hsg->timer_freq  = timer_freq;
    hsg->prescaler   = prescaler;
    hsg->period      = period;
    hsg->current_angle = SG90_DEFAULT_ANGLE;

    /* 验证 PWM 周期是否接近 20ms (允许 ±10% 误差) */
    uint32_t step_ns = SG90_GetStepNS(hsg);
    uint32_t actual_period_us = (uint32_t)((uint64_t)step_ns * (period + 1U) / 1000U);

    /* 范围: 18000us ~ 22000us 视为合法 */
    if (actual_period_us < 18000U || actual_period_us > 22000U) {
        /* PWM 周期偏差过大，SG90 可能无法正常工作 */
        return HAL_ERROR;
    }

    /* 启动 PWM 输出 */
    HAL_StatusTypeDef status = HAL_TIM_PWM_Start(htim, channel);
    if (status != HAL_OK) {
        return status;
    }

    /* 转到默认角度 (90° 中立位) */
    SG90_SetAngle(hsg, SG90_DEFAULT_ANGLE);

    return HAL_OK;
}

/************************ 角度控制函数 ************************/

/**
 * @brief  STM32H750专用：设置SG90舵机角度
 * @param  hsg: 舵机句柄指针
 * @param  angle: 目标角度 (0~180°)
 * @retval HAL_OK(0) - 成功, HAL_ERROR(1) - 参数异常
 * @note   超出范围的角度会被钳位到 0° 或 180°
 */
HAL_StatusTypeDef SG90_SetAngle(SG90_HandleTypeDef *hsg, uint16_t angle)
{
    if (hsg == (void *)0) {
        return HAL_ERROR;
    }

    /* 钳位到合法范围 */
    angle = SG90_ClampAngle(angle);

    /* 角度转换为 CCR 值 (纯整数运算) */
    uint32_t ccr = SG90_AngleToCCR(hsg, angle);

    /* 写入 CCR 寄存器 */
    __HAL_TIM_SET_COMPARE(hsg->htim, hsg->channel, ccr);

    /* 记录当前角度 */
    hsg->current_angle = angle;

    return HAL_OK;
}

/************************ 角度转换函数 ************************/

/**
 * @brief  STM32H750专用：角度转换为定时器CCR寄存器值 (整数运算)
 * @param  hsg: 舵机句柄指针
 * @param  angle: 目标角度 (0~180°)
 * @retval CCR寄存器值
 * @note   计算公式：CCR = angle * (pulse_range) / 180 + ccr_min
 *         全程整数运算，利用H750硬件除法，无浮点开销
 */
uint32_t SG90_AngleToCCR(const SG90_HandleTypeDef *hsg, uint16_t angle)
{
    if (hsg == (void *)0) return 0U;

    uint32_t step_ns = SG90_GetStepNS(hsg);
    if (step_ns == 0U) return 0U;

    /*
     * 计算逻辑：
     * 1. CCR_min: 对应 0.5ms 脉宽的 CCR 值
     *    ccr_min = 500000ns / step_ns
     * 2. CCR_max: 对应 2.5ms 脉宽的 CCR 值
     *    ccr_max = 2500000ns / step_ns
     * 3. CCR = ccr_min + angle * (ccr_max - ccr_min) / 180
     *
     * 使用 uint64_t 中间变量防止乘法溢出
     */
    uint32_t ccr_min  = 500000U / step_ns;
    uint32_t ccr_max  = 2500000U / step_ns;
    uint32_t ccr_range = ccr_max - ccr_min;

    /* 先乘后除，用 uint64_t 防溢出 */
    uint32_t ccr = ccr_min + (uint32_t)((uint64_t)angle * ccr_range / 180U);

    return ccr;
}

/************************ 微秒脉宽控制函数 ************************/

/**
 * @brief  STM32H750专用：直接设置PWM脉宽 (微秒)，用于精细校准
 * @param  hsg: 舵机句柄指针
 * @param  pulse_us: 目标脉宽 (500~2500us)
 * @retval HAL_OK(0) - 成功, HAL_ERROR(1) - 参数异常
 * @note   用于校准模式下直接控制脉宽，绕过角度转换
 */
HAL_StatusTypeDef SG90_SetPulseMicroseconds(SG90_HandleTypeDef *hsg, uint16_t pulse_us)
{
    if (hsg == (void *)0) return HAL_ERROR;

    /* 脉宽范围钳位 */
    if (pulse_us < SG90_PULSE_MIN_US) pulse_us = SG90_PULSE_MIN_US;
    if (pulse_us > SG90_PULSE_MAX_US) pulse_us = SG90_PULSE_MAX_US;

    uint32_t step_ns = SG90_GetStepNS(hsg);
    if (step_ns == 0U) return HAL_ERROR;

    /* 脉宽 (us) → 纳秒 → CCR */
    uint32_t pulse_ns = (uint32_t)pulse_us * 1000U;
    uint32_t ccr = pulse_ns / step_ns;

    /* 确保 CCR 不超过 ARR */
    if (ccr > hsg->period) ccr = hsg->period;

    __HAL_TIM_SET_COMPARE(hsg->htim, hsg->channel, ccr);

    return HAL_OK;
}

/************************ 查询函数 ************************/

/**
 * @brief  获取舵机当前角度
 * @param  hsg: 舵机句柄指针
 * @retval 当前角度 (0~180°)
 */
uint16_t SG90_GetAngle(const SG90_HandleTypeDef *hsg)
{
    if (hsg == (void *)0) return 0U;
    return hsg->current_angle;
}

/************************ 电机控制函数 ************************/

/**
 * @brief  STM32H750专用：启动PWM输出 (开始驱动舵机)
 * @param  hsg: 舵机句柄指针
 * @retval HAL_OK(0) - 成功, 其他值 - 失败
 */
HAL_StatusTypeDef SG90_Start(SG90_HandleTypeDef *hsg)
{
    if (hsg == (void *)0 || hsg->htim == (void *)0) return HAL_ERROR;
    return HAL_TIM_PWM_Start(hsg->htim, hsg->channel);
}

/**
 * @brief  STM32H750专用：停止PWM输出 (舵机断电，失去保持力矩)
 * @param  hsg: 舵机句柄指针
 * @retval HAL_OK(0) - 成功, 其他值 - 失败
 * @note   停止后舵机无保持力矩，外部力可转动舵盘
 */
HAL_StatusTypeDef SG90_Stop(SG90_HandleTypeDef *hsg)
{
    if (hsg == (void *)0 || hsg->htim == (void *)0) return HAL_ERROR;
    return HAL_TIM_PWM_Stop(hsg->htim, hsg->channel);
}
