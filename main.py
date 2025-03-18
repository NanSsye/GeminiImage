import os
import json
import tomllib
import traceback
import uuid
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from loguru import logger
import aiohttp
from PIL import Image

from WechatAPI import WechatAPIClient
from database.XYBotDB import XYBotDB
from utils.decorators import *
from utils.plugin_base import PluginBase


class GeminiImage(PluginBase):
    """基于Google Gemini的图像生成插件"""
    
    description = "基于Google Gemini的图像生成插件"
    author = "老夏的金库"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        
        try:
            # 读取配置
            config_path = os.path.join(os.path.dirname(__file__), "config.toml")
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
                
            # 获取Gemini配置
            plugin_config = config.get("GeminiImage", {})
            self.enable = plugin_config.get("enable", False)
            self.api_key = plugin_config.get("gemini_api_key", "")
            self.model = plugin_config.get("model", "gemini-2.0-flash-exp-image-generation")
            
            # 获取命令配置
            self.commands = plugin_config.get("commands", ["#生成图片", "#画图", "#图片生成"])
            self.edit_commands = plugin_config.get("edit_commands", ["#编辑图片", "#修改图片"])
            self.exit_commands = plugin_config.get("exit_commands", ["#结束对话", "#退出对话", "#关闭对话", "#结束"])  # 从配置读取结束对话命令
            
            # 获取积分配置
            self.enable_points = plugin_config.get("enable_points", True)
            self.generate_cost = plugin_config.get("generate_image_cost", 10)
            self.edit_cost = plugin_config.get("edit_image_cost", 15)
            
            # 获取图片保存配置
            self.save_path = plugin_config.get("save_path", "temp")
            self.save_dir = os.path.join(os.path.dirname(__file__), self.save_path)
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 获取管理员列表
            self.admins = plugin_config.get("admins", [])
            
            # 获取代理配置
            self.enable_proxy = plugin_config.get("enable_proxy", False)
            self.proxy_url = plugin_config.get("proxy_url", "")
            
            # 初始化数据库
            self.db = XYBotDB()
            
            # 初始化会话状态，用于保存上下文
            self.conversations = defaultdict(list)  # 用户ID -> 对话历史列表
            self.conversation_expiry = 600  # 会话过期时间(秒)
            self.conversation_timestamps = {}  # 用户ID -> 最后活动时间
            
            # 存储最后一次生成的图片路径
            self.last_images = {}  # 会话标识 -> 最后一次生成的图片路径
            
            # 全局图片缓存，用于存储最近接收到的图片
            # 修改为使用(聊天ID, 用户ID)作为键，以区分群聊中不同用户
            self.image_cache = {}  # (聊天ID, 用户ID) -> {content: bytes, timestamp: float}
            self.image_cache_timeout = 300  # 图片缓存过期时间(秒)
            
            # 验证关键配置
            if not self.api_key:
                logger.warning("GeminiImage插件未配置API密钥")
                
            logger.info("GeminiImage插件初始化成功")
            if self.enable_proxy:
                logger.info(f"GeminiImage插件已启用代理: {self.proxy_url}")
            
        except Exception as e:
            logger.error(f"GeminiImage插件初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.enable = False
    
    @on_text_message(priority=30)
    async def handle_generate_image(self, bot: WechatAPIClient, message: dict) -> bool:
        """处理生成图片的命令"""
        if not self.enable:
            return True  # 插件未启用，继续执行后续插件
        
        content = message.get("Content", "").strip()
        from_wxid = message.get("FromWxid", "")
        sender_wxid = message.get("SenderWxid", "")
        
        # 清理过期的会话
        self._cleanup_expired_conversations()
        
        # 会话标识
        conversation_key = f"{from_wxid}_{sender_wxid}"
        
        # 检查是否是结束对话命令
        if content in self.exit_commands:
            if conversation_key in self.conversations:
                # 清除会话数据
                del self.conversations[conversation_key]
                if conversation_key in self.conversation_timestamps:
                    del self.conversation_timestamps[conversation_key]
                if conversation_key in self.last_images:
                    del self.last_images[conversation_key]
                
                await bot.send_at_message(from_wxid, "\n已结束Gemini图像生成对话，下次需要时请使用命令重新开始", [sender_wxid])
                return False  # 阻止后续插件执行
            else:
                # 没有活跃会话
                await bot.send_at_message(from_wxid, "\n您当前没有活跃的Gemini图像生成对话", [sender_wxid])
                return False  # 阻止后续插件执行
        
        # 检查是否是生成图片命令
        for cmd in self.commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    await bot.send_at_message(from_wxid, "\n请提供描述内容，格式：#生成图片 [描述]", [sender_wxid])
                    return False  # 命令格式错误，阻止后续插件执行
                
                # 检查API密钥是否配置
                if not self.api_key:
                    await bot.send_at_message(from_wxid, "\n请先在配置文件中设置Gemini API密钥", [sender_wxid])
                    return False
                
                # 检查积分
                if self.enable_points and sender_wxid not in self.admins:
                    points = self.db.get_points(sender_wxid)
                    if points < self.generate_cost:
                        await bot.send_at_message(from_wxid, f"\n您的积分不足，生成图片需要{self.generate_cost}积分，您当前有{points}积分", [sender_wxid])
                        return False  # 积分不足，阻止后续插件执行
                
                # 生成图片
                try:
                    # 发送处理中消息
                    await bot.send_at_message(from_wxid, "\n正在生成图片，请稍候...", [sender_wxid])
                    
                    # 获取上下文历史
                    conversation_history = self.conversations[conversation_key]
                    
                    # 添加用户提示到会话
                    user_message = {"role": "user", "parts": [{"text": prompt}]}
                    
                    # 调用Gemini API生成图片
                    image_data, text_response = await self._generate_image(prompt, conversation_history)
                    
                    if image_data:
                        # 保存图片到本地
                        image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        
                        # 保存最后生成的图片路径
                        self.last_images[conversation_key] = image_path
                        
                        # 扣除积分
                        if self.enable_points and sender_wxid not in self.admins:
                            self.db.add_points(sender_wxid, -self.generate_cost)
                            points_msg = f"已扣除{self.generate_cost}积分，当前剩余{points - self.generate_cost}积分"
                        else:
                            points_msg = ""
                        
                        # 发送文本回复（如果有）
                        if text_response:
                            await bot.send_text_message(from_wxid, f"{text_response}\n\n{points_msg if points_msg else ''}")
                        else:
                            await bot.send_text_message(from_wxid, f"图片生成成功！{points_msg if points_msg else ''}")
                        
                        # 发送图片
                        with open(image_path, "rb") as f:
                            await bot.send_image_message(from_wxid, f.read())
                        
                        # 提示可以结束对话
                        if not conversation_history:  # 如果是新会话
                            await bot.send_text_message(from_wxid, f"已开始图像对话，可以直接发消息继续修改图片。需要结束时请发送\"{self.exit_commands[0]}\"")
                        
                        # 更新会话历史
                        conversation_history.append(user_message)
                        assistant_message = {
                            "role": "model", 
                            "parts": [
                                {"text": text_response if text_response else "我已生成了图片"},
                                {"image_url": image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)
                        
                        # 限制会话历史长度
                        if len(conversation_history) > 10:  # 保留最近5轮对话
                            conversation_history = conversation_history[-10:]
                        
                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        if text_response:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            translated_response = self._translate_gemini_message(text_response)
                            await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                        else:
                            await bot.send_at_message(from_wxid, "\n图片生成失败，请稍后再试或修改提示词", [sender_wxid])
                except Exception as e:
                    logger.error(f"生成图片失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    await bot.send_at_message(from_wxid, f"\n生成图片失败: {str(e)}", [sender_wxid])
                return False  # 已处理命令，阻止后续插件执行
        
        # 检查是否是编辑图片命令（针对已保存的图片）
        for cmd in self.edit_commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    await bot.send_at_message(from_wxid, "\n请提供编辑描述，格式：#编辑图片 [描述]", [sender_wxid])
                    return False  # 命令格式错误，阻止后续插件执行
                
                # 检查API密钥是否配置
                if not self.api_key:
                    await bot.send_at_message(from_wxid, "\n请先在配置文件中设置Gemini API密钥", [sender_wxid])
                    return False
                
                # 先尝试从缓存获取最近的图片
                image_data = await self._get_recent_image(from_wxid, sender_wxid)
                if image_data:
                    # 如果找到缓存的图片，保存到本地再处理
                    image_path = os.path.join(self.save_dir, f"temp_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    self.last_images[conversation_key] = image_path
                    logger.info(f"找到最近缓存的图片，保存到：{image_path}")
                
                # 检查是否有上一次上传/生成的图片
                last_image_path = self.last_images.get(conversation_key)
                if not last_image_path or not os.path.exists(last_image_path):
                    await bot.send_at_message(from_wxid, "\n未找到可编辑的图片，请先上传一张图片", [sender_wxid])
                    return False
                
                # 检查积分
                if self.enable_points and sender_wxid not in self.admins:
                    points = self.db.get_points(sender_wxid)
                    if points < self.edit_cost:
                        await bot.send_at_message(from_wxid, f"\n您的积分不足，编辑图片需要{self.edit_cost}积分，您当前有{points}积分", [sender_wxid])
                        return False  # 积分不足，阻止后续插件执行
                
                # 编辑图片
                try:
                    # 发送处理中消息
                    await bot.send_at_message(from_wxid, "\n正在编辑图片，请稍候...", [sender_wxid])
                    
                    # 读取上一次的图片
                    with open(last_image_path, "rb") as f:
                        image_data = f.read()
                    
                    # 获取会话上下文
                    conversation_history = self.conversations[conversation_key]
                    
                    # 调用Gemini API编辑图片
                    result_image, text_response = await self._edit_image(prompt, image_data, conversation_history)
                    
                    if result_image:
                        # 保存编辑后的图片
                        edited_image_path = os.path.join(self.save_dir, f"edited_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(edited_image_path, "wb") as f:
                            f.write(result_image)
                        
                        # 更新最后生成的图片路径
                        self.last_images[conversation_key] = edited_image_path
                        
                        # 扣除积分
                        if self.enable_points and sender_wxid not in self.admins:
                            self.db.add_points(sender_wxid, -self.edit_cost)
                            points_msg = f"已扣除{self.edit_cost}积分，当前剩余{points - self.edit_cost}积分"
                        else:
                            points_msg = ""
                        
                        # 发送文本回复（如果有）
                        if text_response:
                            await bot.send_text_message(from_wxid, f"{text_response}\n\n{points_msg if points_msg else ''}")
                        else:
                            await bot.send_text_message(from_wxid, f"图片编辑成功！{points_msg if points_msg else ''}")
                        
                        # 发送图片
                        with open(edited_image_path, "rb") as f:
                            await bot.send_image_message(from_wxid, f.read())
                        
                        # 提示可以结束对话
                        if not conversation_history:  # 如果是新会话
                            await bot.send_text_message(from_wxid, f"已开始图像对话，可以直接发消息继续修改图片。需要结束时请发送\"{self.exit_commands[0]}\"")
                        
                        # 更新会话历史
                        user_message = {
                            "role": "user", 
                            "parts": [
                                {"text": prompt},
                                {"image_url": last_image_path}
                            ]
                        }
                        conversation_history.append(user_message)
                        
                        assistant_message = {
                            "role": "model", 
                            "parts": [
                                {"text": text_response if text_response else "我已编辑完成图片"},
                                {"image_url": edited_image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)
                        
                        # 限制会话历史长度
                        if len(conversation_history) > 10:  # 保留最近5轮对话
                            conversation_history = conversation_history[-10:]
                        
                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        if text_response:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            translated_response = self._translate_gemini_message(text_response)
                            await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                            logger.warning(f"API拒绝编辑图片，提示: {text_response}")
                        else:
                            logger.error(f"编辑图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(from_wxid, "\n图片编辑失败，请稍后再试或修改描述", [sender_wxid])
                except Exception as e:
                    logger.error(f"编辑图片失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    await bot.send_at_message(from_wxid, f"\n编辑图片失败: {str(e)}", [sender_wxid])
                return False  # 已处理命令，阻止后续插件执行
        
        # 检查是否是对话继续（没有前缀命令，但有活跃会话）
        if conversation_key in self.conversations and content and not any(content.startswith(cmd) for cmd in self.commands + self.edit_commands):
            # 有活跃会话，且不是其他命令，视为继续对话
            try:
                logger.info(f"继续对话: 用户={sender_wxid}, 内容='{content}'")
                
                # 检查积分
                if self.enable_points and sender_wxid not in self.admins:
                    points = self.db.get_points(sender_wxid)
                    if points < self.generate_cost:
                        await bot.send_at_message(from_wxid, f"\n您的积分不足，生成图片需要{self.generate_cost}积分，您当前有{points}积分", [sender_wxid])
                        return False  # 积分不足，阻止后续插件执行
                
                # 发送处理中消息
                await bot.send_at_message(from_wxid, "\n正在处理您的请求，请稍候...", [sender_wxid])
                
                # 获取上下文历史
                conversation_history = self.conversations[conversation_key]
                logger.info(f"对话历史长度: {len(conversation_history)}")
                
                # 添加用户提示到会话
                user_message = {"role": "user", "parts": [{"text": content}]}
                
                # 检查是否有上一次生成的图片，如果有则自动作为输入
                last_image_path = self.last_images.get(conversation_key)
                logger.info(f"上一次图片路径: {last_image_path}")
                
                if last_image_path and os.path.exists(last_image_path):
                    logger.info(f"找到上一次图片，将使用该图片进行编辑")
                    # 读取上一次生成的图片
                    with open(last_image_path, "rb") as f:
                        image_data = f.read()
                    
                    # 调用编辑图片API
                    logger.info(f"调用编辑图片API")
                    result_image, text_response = await self._edit_image(content, image_data, conversation_history)
                    
                    if result_image:
                        logger.info(f"成功获取编辑后的图片结果")
                        # 保存编辑后的图片
                        new_image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(new_image_path, "wb") as f:
                            f.write(result_image)
                        
                        # 更新最后生成的图片路径
                        self.last_images[conversation_key] = new_image_path
                        
                        # 扣除积分
                        if self.enable_points and sender_wxid not in self.admins:
                            self.db.add_points(sender_wxid, -self.edit_cost)  # 使用编辑积分
                            points_msg = f"已扣除{self.edit_cost}积分，当前剩余{points - self.edit_cost}积分"
                        else:
                            points_msg = ""
                        
                        # 发送文本回复（如果有）
                        if text_response:
                            await bot.send_text_message(from_wxid, f"{text_response}\n\n{points_msg if points_msg else ''}")
                        else:
                            await bot.send_text_message(from_wxid, f"图片编辑成功！{points_msg if points_msg else ''}")
                        
                        # 发送图片
                        logger.info(f"发送编辑后的图片")
                        with open(new_image_path, "rb") as f:
                            await bot.send_image_message(from_wxid, f.read())
                        
                        # 更新会话历史
                        # 添加包含图片的用户消息
                        user_message = {
                            "role": "user", 
                            "parts": [
                                {"text": content},
                                {"image_url": last_image_path}
                            ]
                        }
                        conversation_history.append(user_message)
                        
                        assistant_message = {
                            "role": "model", 
                            "parts": [
                                {"text": text_response if text_response else "我已编辑了图片"},
                                {"image_url": new_image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)
                        
                        # 限制会话历史长度
                        if len(conversation_history) > 10:
                            conversation_history = conversation_history[-10:]
                        
                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                        
                        return False  # 已处理命令，阻止后续插件执行
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        if text_response:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            translated_response = self._translate_gemini_message(text_response)
                            await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                            logger.warning(f"API拒绝编辑图片，提示: {text_response}")
                        else:
                            logger.error(f"编辑图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(from_wxid, "\n图片编辑失败，请稍后再试或修改描述", [sender_wxid])
                else:
                    logger.info(f"没有找到上一次图片或文件不存在，将生成新图片")
                    # 没有上一次图片，当作生成新图片处理
                    image_data, text_response = await self._generate_image(content, conversation_history)
                    
                    if image_data:
                        logger.info(f"成功获取生成的图片结果")
                        # 保存图片到本地
                        image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        
                        # 更新最后生成的图片路径
                        self.last_images[conversation_key] = image_path
                        
                        # 扣除积分
                        if self.enable_points and sender_wxid not in self.admins:
                            self.db.add_points(sender_wxid, -self.generate_cost)
                            points_msg = f"已扣除{self.generate_cost}积分，当前剩余{points - self.generate_cost}积分"
                        else:
                            points_msg = ""
                        
                        # 发送文本回复（如果有）
                        if text_response:
                            await bot.send_text_message(from_wxid, f"{text_response}\n\n{points_msg if points_msg else ''}")
                        else:
                            await bot.send_text_message(from_wxid, f"图片生成成功！{points_msg if points_msg else ''}")
                        
                        # 发送图片
                        logger.info(f"发送生成的图片")
                        with open(image_path, "rb") as f:
                            await bot.send_image_message(from_wxid, f.read())
                        
                        # 更新会话历史
                        conversation_history.append(user_message)
                        assistant_message = {
                            "role": "model", 
                            "parts": [
                                {"text": text_response if text_response else "我已基于您的提示生成了图片"},
                                {"image_url": image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)
                        
                        # 限制会话历史长度
                        if len(conversation_history) > 10:
                            conversation_history = conversation_history[-10:]
                        
                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        if text_response:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            translated_response = self._translate_gemini_message(text_response)
                            await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                            logger.warning(f"API拒绝生成图片，提示: {text_response}")
                        else:
                            logger.error(f"生成图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(from_wxid, "\n图片生成失败，请稍后再试或修改提示词", [sender_wxid])
                return False
            except Exception as e:
                logger.error(f"对话继续生成图片失败: {str(e)}")
                logger.error(traceback.format_exc())
                await bot.send_at_message(from_wxid, f"\n生成失败: {str(e)}", [sender_wxid])
                return False  # 已处理命令，阻止后续插件执行
        
        # 不是本插件的命令，继续执行后续插件
        return True
    
    @on_file_message(priority=30)
    async def handle_edit_image(self, bot: WechatAPIClient, message: dict) -> bool:
        """处理编辑图片的命令"""
        if not self.enable:
            return True  # 插件未启用，继续执行后续插件
        
        from_wxid = message.get("FromWxid", "")
        sender_wxid = message.get("SenderWxid", "")
        file_info = message.get("FileInfo", {})
        
        # 清理过期的会话
        self._cleanup_expired_conversations()
        
        # 会话标识
        conversation_key = f"{from_wxid}_{sender_wxid}"
        
        # 检查消息是否含有文件信息
        if not file_info or "FileID" not in file_info:
            return True  # 不是有效的文件消息，继续执行后续插件
        
        # 检查是否是图片编辑命令
        if "FileSummary" in file_info:
            summary = file_info.get("FileSummary", "").strip()
            
            for cmd in self.edit_commands:
                if summary.startswith(cmd):
                    # 提取提示词
                    prompt = summary[len(cmd):].strip()
                    if not prompt:
                        await bot.send_at_message(from_wxid, "\n请提供编辑描述，格式：#编辑图片 [描述]", [sender_wxid])
                        return False  # 命令格式错误，阻止后续插件执行
                    
                    # 检查API密钥是否配置
                    if not self.api_key:
                        await bot.send_at_message(from_wxid, "\n请先在配置文件中设置Gemini API密钥", [sender_wxid])
                        return False
                    
                    # 检查文件类型是否为图片
                    file_name = file_info.get("FileName", "").lower()
                    valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
                    is_image = any(file_name.endswith(ext) for ext in valid_extensions)
                    
                    if not is_image:
                        await bot.send_at_message(from_wxid, "\n请上传图片文件（支持JPG、PNG、WEBP格式）", [sender_wxid])
                        return False
                    
                    # 检查积分
                    if self.enable_points and sender_wxid not in self.admins:
                        points = self.db.get_points(sender_wxid)
                        if points < self.edit_cost:
                            await bot.send_at_message(from_wxid, f"\n您的积分不足，编辑图片需要{self.edit_cost}积分，您当前有{points}积分", [sender_wxid])
                            return False  # 积分不足，阻止后续插件执行
                    
                    # 编辑图片
                    try:
                        # 发送处理中消息
                        await bot.send_at_message(from_wxid, "\n正在编辑图片，请稍候...", [sender_wxid])
                        
                        # 下载用户上传的图片
                        file_id = file_info.get("FileID")
                        file_content = await bot.download_file(file_id)
                        
                        # 保存原始图片
                        orig_image_path = os.path.join(self.save_dir, f"orig_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(orig_image_path, "wb") as f:
                            f.write(file_content)
                        
                        # 获取会话上下文
                        conversation_history = self.conversations[conversation_key]
                        
                        # 调用Gemini API编辑图片
                        image_data, text_response = await self._edit_image(prompt, file_content, conversation_history)
                        
                        if image_data:
                            # 保存编辑后的图片
                            edited_image_path = os.path.join(self.save_dir, f"edited_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                            with open(edited_image_path, "wb") as f:
                                f.write(image_data)
                            
                            # 更新最后生成的图片路径
                            self.last_images[conversation_key] = edited_image_path
                            
                            # 扣除积分
                            if self.enable_points and sender_wxid not in self.admins:
                                self.db.add_points(sender_wxid, -self.edit_cost)
                                points_msg = f"已扣除{self.edit_cost}积分，当前剩余{points - self.edit_cost}积分"
                            else:
                                points_msg = ""
                            
                            # 发送文本回复（如果有）
                            if text_response:
                                await bot.send_text_message(from_wxid, f"{text_response}\n\n{points_msg if points_msg else ''}")
                            else:
                                await bot.send_text_message(from_wxid, f"图片编辑成功！{points_msg if points_msg else ''}")
                            
                            # 发送图片
                            with open(edited_image_path, "rb") as f:
                                await bot.send_image_message(from_wxid, f.read())
                            
                            # 提示可以结束对话
                            if not conversation_history:  # 如果是新会话
                                await bot.send_text_message(from_wxid, f"已开始图像对话，可以直接发消息继续修改图片。需要结束时请发送\"{self.exit_commands[0]}\"")
                            
                            # 更新会话历史
                            user_message = {
                                "role": "user", 
                                "parts": [
                                    {"text": prompt},
                                    {"image_url": orig_image_path}
                                ]
                            }
                            conversation_history.append(user_message)
                            
                            assistant_message = {
                                "role": "model", 
                                "parts": [
                                    {"text": text_response if text_response else "我已编辑完成图片"},
                                    {"image_url": edited_image_path}
                                ]
                            }
                            conversation_history.append(assistant_message)
                            
                            # 限制会话历史长度
                            if len(conversation_history) > 10:  # 保留最近5轮对话
                                conversation_history = conversation_history[-10:]
                            
                            # 更新会话时间戳
                            self.conversation_timestamps[conversation_key] = time.time()
                        else:
                            # 检查是否有文本响应，可能是内容被拒绝
                            if text_response:
                                # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                                translated_response = self._translate_gemini_message(text_response)
                                await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                                logger.warning(f"API拒绝编辑图片，提示: {text_response}")
                            else:
                                logger.error(f"编辑图片失败，未获取到有效的图片数据")
                                await bot.send_at_message(from_wxid, "\n图片编辑失败，请稍后再试或修改描述", [sender_wxid])
                    except Exception as e:
                        logger.error(f"编辑图片失败: {str(e)}")
                        logger.error(traceback.format_exc())
                        await bot.send_at_message(from_wxid, f"\n编辑图片失败: {str(e)}", [sender_wxid])
                    return False  # 已处理命令，阻止后续插件执行
        
        # 不是本插件的命令，继续执行后续插件
        return True
    
    @on_image_message(priority=30)
    async def handle_image_edit(self, bot: WechatAPIClient, message: dict) -> bool:
        """处理图片消息，缓存图片数据以备后续编辑使用"""
        if not self.enable:
            return True  # 插件未启用，继续执行后续插件
        
        from_wxid = message.get("FromWxid", "")
        sender_wxid = message.get("SenderWxid", "")
        
        # 在群聊中，使用发送者ID作为图片所有者
        # 在私聊中，FromWxid和SenderWxid相同
        is_group = message.get("IsGroup", False)
        image_owner = sender_wxid if is_group else from_wxid
        
        try:
            # 清理过期缓存
            self._cleanup_image_cache()
            
            # 提取图片数据 - 首先尝试直接从ImgBuf获取
            if "ImgBuf" in message and message["ImgBuf"] and len(message["ImgBuf"]) > 100:
                image_data = message["ImgBuf"]
                logger.info(f"从ImgBuf提取到图片数据，大小: {len(image_data)} 字节")
                
                # 保存图片到缓存 - 使用(聊天ID, 用户ID)作为键
                cache_key = (from_wxid, image_owner)
                self.image_cache[cache_key] = {
                    "content": image_data,
                    "timestamp": time.time()
                }
                return True
            
            # 如果ImgBuf中没有有效数据，尝试从Content中提取Base64图片数据
            content = message.get("Content", "")
            if content and isinstance(content, str):
                # 检查是否是XML格式
                if content.startswith("<?xml") and "<img" in content:
                    logger.info("检测到XML格式图片消息，尝试提取Base64数据")
                    try:
                        # 查找XML后附带的Base64数据
                        xml_end = content.find("</msg>")
                        if xml_end > 0 and len(content) > xml_end + 6:
                            # XML后面可能有Base64数据
                            base64_data = content[xml_end + 6:].strip()
                            if base64_data:
                                try:
                                    image_data = base64.b64decode(base64_data)
                                    logger.info(f"从XML后面提取到Base64数据，长度: {len(image_data)} 字节")
                                    
                                    # 保存图片到缓存 - 使用(聊天ID, 用户ID)作为键
                                    cache_key = (from_wxid, image_owner)
                                    self.image_cache[cache_key] = {
                                        "content": image_data,
                                        "timestamp": time.time()
                                    }
                                    return True
                                except Exception as e:
                                    logger.error(f"XML后Base64解码失败: {e}")
                        
                        # 如果上面的方法失败，尝试直接检测任何位置的Base64图片头部标识
                        base64_markers = ["iVBOR", "/9j/", "R0lGOD", "UklGR", "PD94bWw", "Qk0", "SUkqAA"]
                        for marker in base64_markers:
                            if marker in content:
                                idx = content.find(marker)
                                if idx > 0:
                                    try:
                                        # 可能的Base64数据，截取从标记开始到结束的部分
                                        base64_data = content[idx:]
                                        # 去除可能的非Base64字符
                                        base64_data = ''.join(c for c in base64_data if c in 
                                                              'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
                                        
                                        # 修正长度确保是4的倍数
                                        padding = len(base64_data) % 4
                                        if padding:
                                            base64_data += '=' * (4 - padding)
                                        
                                        # 尝试解码
                                        image_data = base64.b64decode(base64_data)
                                        if len(image_data) > 1000:  # 确保至少有一些数据
                                            logger.info(f"从内容中提取到{marker}格式图片数据，长度: {len(image_data)} 字节")
                                            
                                            # 保存图片到缓存 - 使用(聊天ID, 用户ID)作为键
                                            cache_key = (from_wxid, image_owner)
                                            self.image_cache[cache_key] = {
                                                "content": image_data,
                                                "timestamp": time.time()
                                            }
                                            return True
                                    except Exception as e:
                                        logger.error(f"提取{marker}格式图片数据失败: {e}")
                    except Exception as e:
                        logger.error(f"提取XML中图片数据失败: {e}")
                        
                # 如果前面的方法都失败了，再尝试一种方法，直接提取整个content作为可能的Base64数据
                # 这对于某些不标准的消息格式可能有效
                try:
                    # 尝试将整个content作为Base64处理
                    base64_content = content.replace(' ', '+')  # 修复可能的URL安全编码
                    # 修正长度确保是4的倍数
                    padding = len(base64_content) % 4
                    if padding:
                        base64_content += '=' * (4 - padding)
                    
                    image_data = base64.b64decode(base64_content)
                    # 如果解码成功且数据量足够大，可能是图片
                    if len(image_data) > 10000:  # 图片数据通常较大
                        try:
                            # 仅尝试打开，不进行验证，避免某些非标准图片格式失败
                            with Image.open(BytesIO(image_data)) as img:
                                width, height = img.size
                                if width > 10 and height > 10:  # 确保是有效图片
                                    logger.info(f"从内容作为Base64解码成功，图片尺寸: {width}x{height}")
                                    
                                    # 保存图片到缓存 - 使用(聊天ID, 用户ID)作为键
                                    cache_key = (from_wxid, image_owner)
                                    self.image_cache[cache_key] = {
                                        "content": image_data,
                                        "timestamp": time.time()
                                    }
                                    return True
                        except Exception as img_e:
                            logger.error(f"解码后数据不是有效图片: {img_e}")
                except Exception as e:
                    # 解码失败不是错误，只是这种方法不适用
                    pass
            
            logger.warning("未能从消息中提取有效的图片数据")
        except Exception as e:
            logger.error(f"处理图片消息失败: {str(e)}")
            logger.error(traceback.format_exc())
        
        return True  # 继续执行后续插件
    
    def _cleanup_expired_conversations(self):
        """清理过期的会话"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.conversation_timestamps.items():
            if current_time - timestamp > self.conversation_expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.conversations:
                del self.conversations[key]
            if key in self.conversation_timestamps:
                del self.conversation_timestamps[key]
            if key in self.last_images:
                del self.last_images[key]
    
    async def _generate_image(self, prompt: str, conversation_history: List[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """调用Gemini API生成图片，返回图片数据和文本响应"""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": self.api_key
        }
        
        # 构建请求数据
        if conversation_history and len(conversation_history) > 0:
            # 有会话历史，构建上下文
            # 需要处理会话历史中的图片格式
            processed_history = []
            for msg in conversation_history:
                # 转换角色名称，确保使用 "user" 或 "model"
                role = msg["role"]
                if role == "assistant":
                    role = "model"
                
                processed_msg = {"role": role, "parts": []}
                for part in msg["parts"]:
                    if "text" in part:
                        processed_msg["parts"].append({"text": part["text"]})
                    elif "image_url" in part:
                        # 需要读取图片并转换为inlineData格式
                        try:
                            with open(part["image_url"], "rb") as f:
                                image_data = f.read()
                                image_base64 = base64.b64encode(image_data).decode("utf-8")
                                processed_msg["parts"].append({
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": image_base64
                                    }
                                })
                        except Exception as e:
                            logger.error(f"处理历史图片失败: {e}")
                            # 跳过这个图片
                processed_history.append(processed_msg)
            
            data = {
                "contents": processed_history + [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        else:
            # 无会话历史，直接使用提示
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        
        # 创建代理配置
        proxy = None
        if self.enable_proxy and self.proxy_url:
            proxy = self.proxy_url
        
        try:
            # 创建客户端会话，设置代理（如果启用）
            async with aiohttp.ClientSession() as session:
                try:
                    # 使用代理发送请求
                    logger.info(f"开始调用Gemini API生成图片")
                    async with session.post(
                        url, 
                        headers=headers, 
                        params=params, 
                        json=data,
                        proxy=proxy,
                        timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                    ) as response:
                        response_text = await response.text()
                        logger.info(f"Gemini API响应状态码: {response.status}")
                        
                        if response.status == 200:
                            try:
                                result = json.loads(response_text)
                                
                                # 记录完整响应内容，方便调试
                                logger.info(f"Gemini API响应内容: {response_text}")
                                
                                # 提取响应
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])
                                    
                                    # 处理文本和图片响应
                                    text_response = None
                                    image_data = None
                                    
                                    for part in parts:
                                        # 处理文本部分
                                        if "text" in part and part["text"]:
                                            text_response = part["text"]
                                        
                                        # 处理图片部分
                                        if "inlineData" in part:
                                            inline_data = part.get("inlineData", {})
                                            if inline_data and "data" in inline_data:
                                                # 返回Base64解码后的图片数据
                                                image_data = base64.b64decode(inline_data["data"])
                                    
                                    if not image_data:
                                        logger.error(f"API响应中没有找到图片数据: {result}")
                                    
                                    return image_data, text_response
                                
                                logger.error(f"未找到生成的图片数据: {result}")
                                return None, None
                            except json.JSONDecodeError as je:
                                logger.error(f"解析JSON响应失败: {je}")
                                logger.error(f"响应内容: {response_text[:1000]}...")  # 记录部分响应内容
                                return None, None
                        else:
                            logger.error(f"Gemini API调用失败 (状态码: {response.status}): {response_text}")
                            return None, None
                except aiohttp.ClientError as ce:
                    logger.error(f"API请求客户端错误: {ce}")
                    return None, None
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None
    
    async def _edit_image(self, prompt: str, image_data: bytes, conversation_history: List[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """调用Gemini API编辑图片，返回图片数据和文本响应"""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": self.api_key
        }
        
        # 将图片数据转换为Base64编码
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # 构建请求数据
        if conversation_history and len(conversation_history) > 0:
            # 有会话历史，构建上下文
            # 需要处理会话历史中的图片格式
            processed_history = []
            for msg in conversation_history:
                # 转换角色名称，确保使用 "user" 或 "model"
                role = msg["role"]
                if role == "assistant":
                    role = "model"
                
                processed_msg = {"role": role, "parts": []}
                for part in msg["parts"]:
                    if "text" in part:
                        processed_msg["parts"].append({"text": part["text"]})
                    elif "image_url" in part:
                        # 需要读取图片并转换为inlineData格式
                        try:
                            with open(part["image_url"], "rb") as f:
                                img_data = f.read()
                                img_base64 = base64.b64encode(img_data).decode("utf-8")
                                processed_msg["parts"].append({
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": img_base64
                                    }
                                })
                        except Exception as e:
                            logger.error(f"处理历史图片失败: {e}")
                            # 跳过这个图片
                processed_history.append(processed_msg)

            data = {
                "contents": processed_history + [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        else:
            # 无会话历史，直接使用提示和图片
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        
        # 创建代理配置
        proxy = None
        if self.enable_proxy and self.proxy_url:
            proxy = self.proxy_url
        
        try:
            # 创建客户端会话，设置代理（如果启用）
            async with aiohttp.ClientSession() as session:
                try:
                    # 使用代理发送请求
                    logger.info(f"开始调用Gemini API编辑图片")
                    async with session.post(
                        url, 
                        headers=headers, 
                        params=params, 
                        json=data,
                        proxy=proxy,
                        timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                    ) as response:
                        response_text = await response.text()
                        logger.info(f"Gemini API响应状态码: {response.status}")
                        
                        if response.status == 200:
                            try:
                                result = json.loads(response_text)
                                
                                # 记录完整响应内容，方便调试
                                logger.info(f"Gemini API响应内容: {response_text}")
                                
                                # 提取响应
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])
                                    
                                    # 处理文本和图片响应
                                    text_response = None
                                    image_data = None
                                    
                                    for part in parts:
                                        # 处理文本部分
                                        if "text" in part and part["text"]:
                                            text_response = part["text"]
                                        
                                        # 处理图片部分
                                        if "inlineData" in part:
                                            inline_data = part.get("inlineData", {})
                                            if inline_data and "data" in inline_data:
                                                # 返回Base64解码后的图片数据
                                                image_data = base64.b64decode(inline_data["data"])
                                    
                                    if not image_data:
                                        logger.error(f"API响应中没有找到图片数据: {result}")
                                    
                                    return image_data, text_response
                                
                                logger.error(f"未找到编辑后的图片数据: {result}")
                                return None, None
                            except json.JSONDecodeError as je:
                                logger.error(f"解析JSON响应失败: {je}")
                                logger.error(f"响应内容: {response_text[:1000]}...")  # 记录部分响应内容
                                return None, None
                        else:
                            logger.error(f"Gemini API调用失败 (状态码: {response.status}): {response_text}")
                            return None, None
                except aiohttp.ClientError as ce:
                    logger.error(f"API请求客户端错误: {ce}")
                    return None, None
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None
    
    def _translate_gemini_message(self, text: str) -> str:
        """将Gemini API的英文消息翻译成中文"""
        # 常见的内容审核拒绝消息翻译
        if "I'm unable to create this image" in text:
            if "sexually suggestive" in text:
                return "抱歉，我无法创建这张图片。我不能生成带有性暗示或促进有害刻板印象的内容。请提供其他描述。"
            elif "harmful" in text or "dangerous" in text:
                return "抱歉，我无法创建这张图片。我不能生成可能有害或危险的内容。请提供其他描述。"
            elif "violent" in text:
                return "抱歉，我无法创建这张图片。我不能生成暴力或血腥的内容。请提供其他描述。"
            else:
                return "抱歉，我无法创建这张图片。请尝试修改您的描述，提供其他内容。"
        
        # 其他常见拒绝消息
        if "cannot generate" in text or "can't generate" in text:
            return "抱歉，我无法生成符合您描述的图片。请尝试其他描述。"
        
        if "against our content policy" in text:
            return "抱歉，您的请求违反了内容政策，无法生成相关图片。请提供其他描述。"
        
        # 默认情况，原样返回
        return text
    
    def _cleanup_image_cache(self):
        """清理过期的图片缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, cache_data in self.image_cache.items():
            if current_time - cache_data["timestamp"] > self.image_cache_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.image_cache[key]
            
    async def _get_recent_image(self, chat_id: str, user_id: str) -> Optional[bytes]:
        """获取最近的图片数据，区分群聊中的不同用户"""
        # 先尝试从用户专属缓存获取
        cache_key = (chat_id, user_id)
        if cache_key in self.image_cache:
            cache_data = self.image_cache[cache_key]
            if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                logger.info(f"找到用户 {user_id} 在聊天 {chat_id} 中的图片缓存")
                return cache_data["content"]
        
        # 如果是私聊且没找到，尝试使用旧格式的键
        if chat_id == user_id and chat_id in self.image_cache:
            cache_data = self.image_cache[chat_id]
            if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                logger.info(f"找到旧格式的图片缓存，键: {chat_id}")
                return cache_data["content"]
        
        return None 
