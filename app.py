import streamlit as st
from openai import OpenAI


# =========================
# إعداد الصفحة
# =========================

st.set_page_config(
    page_title="TA27A - Agent to Agent",
    page_icon="🤖",
    layout="wide"
)


# =========================
# التصميم
# =========================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .agent-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# العنوان
# =========================

st.markdown(
    '<div class="main-title">🤖 TA27A</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Agent to Agent — منصة متعددة الوكلاء لتحليل المعلومات</div>',
    unsafe_allow_html=True
)


# =========================
# الاتصال بـ DeepSeek
# =========================

try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    api_key = None


# =========================
# الواجهة
# =========================

st.markdown("### 🧠 اكتب المهمة التي تريد من الوكلاء تنفيذها")

user_prompt = st.text_area(
    "المهمة",
    placeholder="مثال: حلل أهمية الذكاء الاصطناعي في الشركات الصغيرة وقدم لي تقريرًا مختصرًا...",
    height=180,
    label_visibility="collapsed"
)


run_button = st.button(
    "🚀 تشغيل نظام A2A",
    use_container_width=True
)


# =========================
# تشغيل الوكلاء
# =========================

if run_button:

    if not api_key:
        st.error(
            "⚠️ مفتاح DeepSeek غير موجود. سنضيفه بأمان في Streamlit Secrets في الخطوة التالية."
        )
        st.stop()

    if not user_prompt.strip():
        st.warning("اكتب المهمة أولًا.")
        st.stop()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # ---------------------------------
    # Agent 1
    # ---------------------------------

    with st.status("🤖 Agent 1 — محلل المعلومات يعمل...", expanded=True):

        response_1 = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": """
أنت Agent متخصص في تحليل المعلومات.
حلل الطلب المقدم إليك بعمق.
استخرج الحقائق والأفكار والنقاط المهمة.
لا تكتب تقريرًا نهائيًا، بل قدم مادة تحليلية دقيقة للوكيل التالي.
"""
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        analysis = response_1.choices[0].message.content

        st.write("تم الانتهاء من التحليل الأولي.")


    # ---------------------------------
    # Agent 2
    # ---------------------------------

    with st.status("🔎 Agent 2 — المراجع والناقد يعمل...", expanded=True):

        response_2 = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": """
أنت Agent ناقد ومراجع.
راجع تحليل Agent السابق.
حدد النقاط القوية والضعيفة.
اكتشف التناقضات أو المعلومات غير الواضحة.
اقترح تحسينات قبل إعداد التقرير النهائي.
"""
                },
                {
                    "role": "user",
                    "content": f"""
المهمة الأصلية:
{user_prompt}

تحليل Agent 1:
{analysis}
"""
                }
            ]
        )

        review = response_2.choices[0].message.content

        st.write("تمت مراجعة التحليل.")


    # ---------------------------------
    # Agent 3
    # ---------------------------------

    with st.status("✍️ Agent 3 — كاتب التقرير يعمل...", expanded=True):

        response_3 = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": """
أنت Agent متخصص في كتابة التقارير التنفيذية.
استخدم التحليل والمراجعة لإنتاج تقرير احترافي.
اجعل التقرير واضحًا ومنظمًا ومباشرًا.
استخدم Markdown والعناوين والنقاط.
اكتب باللغة العربية.
"""
                },
                {
                    "role": "user",
                    "content": f"""
المهمة الأصلية:
{user_prompt}

تحليل Agent 1:
{analysis}

مراجعة Agent 2:
{review}

اكتب التقرير النهائي الآن.
"""
                }
            ]
        )

        final_report = response_3.choices[0].message.content


    # =========================
    # النتيجة
    # =========================

    st.success("✅ اكتمل تنفيذ نظام A2A")

    st.markdown("---")

    st.markdown("## 📄 التقرير النهائي")

    st.markdown(final_report)

    # =========================
    # التفاصيل
    # =========================

    with st.expander("🔍 عرض تحليل Agent 1"):
        st.markdown(analysis)

    with st.expander("🔎 عرض مراجعة Agent 2"):
        st.markdown(review)