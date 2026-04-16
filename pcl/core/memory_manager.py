from schemas.models import StudentProfile

class MemoryManager:
    def __init__(self):
        # 模拟数据库中的学生画像
        self.db = {
            "Student_A": StudentProfile(student_id="Student_A", cog_history="概念薄弱，需要具体例子", aff_history="容易急躁"),
            "Student_B": StudentProfile(student_id="Student_B", soc_history="经常搭便车，需要直接点名邀请")
        }
        
    def get_profiles(self, student_ids: list) -> dict:
        return {sid: self.db.get(sid, StudentProfile(student_id=sid)) for sid in student_ids}

    def update_memory(self, dialog_window, intervention):
        """
        在每轮干预结束后调用大模型总结并更新 JSON 画像。
        (为了代码简洁，此处省略具体 LLM 总结代码，仅留接口)
        """
        pass