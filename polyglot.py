"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        
        ### 템플릿 이름 설정 ###
        
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./Template", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        
        ### input이 존재하면 명령문과 입력 값으로 문자열 생성 ###
        
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        
        ### input이 존재하지 않으면 명령문만 사용해서 문자열 생성 ### 
        
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    

def model_settings():
    peft_model_id = "./config"
    config = PeftConfig.from_pretrained(peft_model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map='auto')
    model = PeftModel.from_pretrained(model, peft_model_id)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)


    # model = AutoModelForCausalLM.from_pretrained('EleutherAI/polyglot-ko-1.3b', device_map='auto')
    # tokenizer = AutoTokenizer.from_pretrained('EleutherAI/polyglot-ko-1.3b')
    model.eval()
    
    return model, tokenizer


## 요약 생성

def gen(model, tokenizer, input) :
    
    inputs = tokenizer(
        f"### 명령어: \n글을 요약해라\n\n### 원본: \n{input}\n\n### 요약: \n",
        return_tensors='pt',
        return_token_type_ids=False
    )
    inputs = inputs.to('cuda')  # 입력 데이터를 CUDA로 이동

    gened = model.generate(
        **inputs,
        max_new_tokens = 512,
        early_stopping=True,
        top_k = 50,
        top_p = 0.95,
        do_sample=True,
        num_return_sequences = 3,
        
        eos_token_id=2,
    )
    

    return tokenizer.decode(gened[0])

    # print(tokenizer.decode(gened[0]))
    # print(tokenizer.decode(gened[1]))
    # print(tokenizer.decode(gened[2]))


# ## 요약해야할 원본 문장

# text = '''
# 산림이나 바다와 같은 자연환경은 마음의 안정감을 주는 등 사람에게 다양한 이점을 준다. 최근에는 이러한 자연환경이 우리 건강에 직접적으로 영향을 미친다는 연구결과가 계속해서 나오고 있다.   기존의 연구들에 따르면 환경오염이 심한 지역에 살면, 치매 등 각종 신경계 질환 발병 위험이 증가하는 것으로 알려졌다. 특히, 미세먼지와 대기 중 높은 수준의 이산화질소 수치는 파킨슨병 발병과 직접적으로 연관이 있다. 실제로 서울아산병원 정선주 신경과 교수와 연구진이 2021년 발표한 연구에 따르면, 대기 중 이산화탄소 수치가 높은 지역에 거주하는 사람은 그렇지 않은 사람보다 파킨슨병에 걸릴 위험이 약 1.14배 높았다. 당시 연구진은 “”체내로 유입된 이산화질소가 염증 반응을 일으켜 뇌에 염증을 유도했거나, 뇌의 미토콘드리아 기능 저하를 유발해 이러한 현상이 나타나는 것 같다“”라고 설명했다.  앞서 설명한 연구를 뒷받침하는 주장으로, 자연친화적 환경에 거주하면 각종 신경계 질환 위험이 감소한다는 의견도 있다. 다만, 그 동안 이를 입증할만한 신뢰성 있는 연구가 미비했다. 미국 하버드 T.H 공중 보건 대학교(Harvard T.H. Chan School of Public Health) 요헴 O. 클롬프메이커(Jochem O. Klompmaker) 박사와 환경위생학과 연구진은 이러한 사실에 착안해 자연환경이 신경계 질환(파킨슨병 및 알츠하이머병)에 미치는 영향에 대한 연구를 실시했다. 이들 연구는 2022년 12월 국제 학술지 ‘자마 네트워크 오픈(JAMA Network Open)’에 게재됐다.연구는 2000년에서 2016년 사이 미국 본토에 거주하면서 미국 노인의료보험의 혜택을 받은 65~74세 노년층 약 6,200만 명을 대상으로 진행됐다. 55%(약 3,410만 명)가 여성이었으며, 대부분이 백인이었다. 조사 기간(16년) 동안 전체 대상자 중 773만 7,609명이 알츠하이머병으로 병원에 입원했으며 116만 8,940명이 파킨슨병으로 입원했다. 연구진은 이 정보를 기반으로 주변 식물의 양 또는 공원 및 수변공간 존재 여부 등을 포함해 연구 대상자의 거주환경과 알츠하이머병 또는 파킨슨병으로 병원에 입원한 횟수 사이의 상관관계를 분석 및 조사했다. 연구 목적은 자연환경이 질병 발병 위험에 미치는 영향보다 자연친화적인 생활·환경이 신경계 질환의 진행을 얼마나 늦추는지에 초점을 맞췄다. 그 결과, 거주지 주변에 나무 등 식물이 많은 사람은 그렇지 않은 사람보다 알츠하이머병으로 병원에 입원할 위험이 더 적다는 사실이 드러났다. 특히, 운동 기능 저하를 동반하는 파킨슨병의 경우 이러한 현상이 더 두드러졌다. 연구 내용을 살펴보면, 거주지 주변 공원 이용이 16% 증가할 때 파킨슨병으로 인한 입원 위험이 3% 감소했으며, 시냇물이나 강 등 수변공간 근처에 사는 사람도 물 근처에 살지 않는 사람보다 입원 위험이 3% 적었다. 클롬프메이커 박사는 “”추가적인 연구가 더 필요하겠지만, 이번 연구를 통해 친환경적인 생활이 신경계 질환 위험을 감소시킨다는 사실이 증명됐다“”라고 밝히며, “”자연환경이 건강에 유익한 영향을 미칠 수 있다“”라고 전했다. 한편, 연구를 평가한 영국 셰필드 대학교(University of Sheffield) 파블로 나바레테 에르난데스(Pablo Navarrete-Hernandez) 박사는 “”집안에 들어오는 햇빛의 양이 많을수록 사람의 행복도가 높아진다“”라고 말하며, “”이와 마찬가지로 자연의 녹색 공간은 사람의 긍정적인 감정을 유발하며, 부정적인 감정을 줄여준다. 이는 스트레스 호르몬인 코르티솔 수치를 낮추고 각종 신경계 질환 위험 감소로 이어진 것 같다“”라고 설명했다. 스트레스 호르몬인 코르티솔이 과다 분비되면, 스트레스를 제어하고 기억을 관리하는 뇌 영역인 해마의 부피가 줄어든다.  자연환경이 신경계 질환 위험 감소에 긍정적인 영향을 미친다는 연구 결과들이 속속들이 나오자, 자연환경을 통해 알츠하이머병 등 치매를 예방 및 치료하고자 하는 시도가 늘어나고 있다. 미국과 핀란드, 일본 등 선진국에서는 오래전부터 치매 치료와 예방에 자연환경과 식물 재배 프로그램을 적극 활용하고 있다. 국내에서도 자연환경을 치매 치료에 활용하고자 하는 움직임이 활발하다. 용인과 진천 등 다양한 지자체에서는 이미 치매 예방 및 치료를 위한 원예, 숲 체험 프로그램을 운영하고 있다. 또한, 관련 연구도 진행되고 있는데 지난 23일 전라남도산림자원연구소는 국립나주병원, 나주시보건소와 합동으로 “”치매 고위험군에 대한 산림치유 효과를 검증하기 위한 연구를 시작한다“”라고 발표했다. 전라남도산림자원연구소는 지난해 직장인 38명을 대상으로 산림치유가 우울증 및 스트레스 완화에 효과가 있다는 연구를 발표한 적 있다. 연구 결과에 따르면, 산림치유 프로그램은 불안 등 부정적인 감정과 체내 코르티솔 수치를 확실히 개선했다. 오득실 전라남도산림자원연구소장은 “”노령인구 증가로 치매 환자가 늘고 있어 치매 예방이 중요한 사회문제로 대두되고 있다“”라고 말하며, “”이번 연구가 치매 전 단계인 고위험군 관리 방안에 대한 정책적 방침 제시에 도움이 될 것으로 기대한다“”라고 전했다.
# '''

# ## 요약 문장

# answer = gen(text)