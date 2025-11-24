// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <string.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <set>
#include "ZmqServer.h"
#include "ZmqClient.h"
#include <cwchar>
#include <locale>
#include <clocale>
#include <cstdlib>
#include <codecvt>
#include <regex>
#include <cwchar>
#include <cstdlib>
#include <codecvt>

using namespace std;

zmq_component::ZmqServer server;
zmq_component::ZmqClient tts_client_("tcp://localhost:7777");

void exit_handler(int signal)
{

    exit(signal);
}

void message_worker(const std::string &rag_text)
{
    static const std::wregex wide_delimiter(
        L"([。！？；：\n]|\\?\\s|\\!\\s|\\；|\\，|\\、|\\|)");
    const std::wstring END_MARKER = L"END";

    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    std::wstring wide_text = converter.from_bytes(rag_text) + END_MARKER;

    std::wsregex_iterator it(wide_text.begin(), wide_text.end(), wide_delimiter);
    std::wsregex_iterator end;

    int skip_counter = 0;
    size_t last_pos = 0;
    while (it != end && skip_counter < 2)
    {
        last_pos = it->position() + it->length();
        ++it;
        ++skip_counter;
    }

    while (it != end)
    {
        size_t seg_start = last_pos;
        size_t seg_end = it->position();
        last_pos = seg_end + it->length();

        std::wstring wide_segment = wide_text.substr(seg_start, seg_end - seg_start);

        wide_segment.erase(0, wide_segment.find_first_not_of(L" \t\n\r"));
        wide_segment.erase(wide_segment.find_last_not_of(L" \t\n\r") + 1);

        if (!wide_segment.empty())
        {
            // 转换回UTF-8
            auto response1 = tts_client_.request(converter.to_bytes(wide_segment));
            std::cout << "[tts -> llm] received: " << response1 << std::endl;
            // queue.push_text(converter.to_bytes(wide_segment));
        }
        ++it;
    }

    // 处理剩余内容
    if (last_pos < wide_text.length())
    {
        std::wstring last_segment = wide_text.substr(last_pos);
        if (!last_segment.empty())
        {
            auto response1 = tts_client_.request(converter.to_bytes(last_segment));
            std::cout << "[tts -> llm] received: " << response1 << std::endl;
            // queue.push_text(converter.to_bytes(last_segment));
        }
    }
}

void receive_asr_data_and_process()
{

    while (true)
    {

        std::string input_str = server.receive();
        std::cout << "[voice -> llm] received: " << input_str << std::endl;
        server.send("llm sucess reply !!!");

        message_worker(input_str);
    }
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "en_US.UTF-8");

    signal(SIGINT, exit_handler);
    printf("rkllm init start\n");

    receive_asr_data_and_process();

    return 0;
}