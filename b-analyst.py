from dataclasses import dataclass
from multiprocessing import context
from typing import Any
from pydantic_ai.models.huggingface import HuggingFaceModel, HuggingFaceModelSettings
from pydantic_ai.providers.huggingface import HuggingFaceProvider
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append, reduce_list_extend, reduce_sum
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import re

load_dotenv("./.env", override=True)

class HFModelSettings(BaseSettings):
    hf_token: str
    hf_code_model_repo: str
    hf_code_model_provider_name: str
    hf_thinking_model_repo: str
    hf_thinking_model_provider_name: str
    temperature: float
    gemini_api_key: str
    gemini_model_name: str 

"""

class CodingAgent(Agent):
    def __init__(self, config: HFModelSettings, output_type: BaseModel):

        model_settings = HuggingFaceModelSettings(
            temperature=config.temperature,
        )

        provider = HuggingFaceProvider(
            api_key=config.hf_token,
            provider_name=config.hf_code_model_provider_name,
        )

        model = HuggingFaceModel(
            model_name=config.hf_code_model_repo,
            provider=provider,
        )

        super().__init__(
            model=model,
            model_settings=model_settings,
            output_type=output_type,
        )

class ThinkingAgent(Agent):
    def __init__(self, config: HFModelSettings, output_type: BaseModel):

        model_settings = HuggingFaceModelSettings(
            temperature=config.temperature,
        )

        provider = HuggingFaceProvider(
            api_key=config.hf_token,
            provider_name=config.hf_thinking_model_provider_name,
        )

        model = HuggingFaceModel(
            model_name=config.hf_thinking_model_repo,
            provider=provider,
        )

        super().__init__(
            model=model,
            model_settings=model_settings,
            output_type=output_type,
        )

"""

class CodingAgent(Agent):
    def __init__(self, config: HFModelSettings, output_type: BaseModel):

        model_settings = GoogleModelSettings(
            temperature=config.temperature,
        )
        provider = GoogleProvider(
            api_key=config.gemini_api_key,
        )
        model = GoogleModel(
            model_name=config.gemini_model_name,
            provider=provider,
        )

        super().__init__(
            model=model,
            model_settings=model_settings,
            output_type=output_type,
        )

class ThinkingAgent(Agent):
    def __init__(self, config: HFModelSettings, output_type: BaseModel):

        model_settings = GoogleModelSettings(
            temperature=config.temperature,
        )
        provider = GoogleProvider(
            api_key=config.gemini_api_key,
        )
        model = GoogleModel(
            model_name=config.gemini_model_name,
            provider=provider,
        )

        super().__init__(
            model=model,
            model_settings=model_settings,
            output_type=output_type,
        )

        

class BusinessMetadata(BaseModel):
    business_name: str = Field(..., description="Name of the business")
    industry: str = Field(default="", description="Industry sector of the business")
    company_size: str = Field(default="", description="Size of the company (e.g., small, medium, large)")
    summary: str = Field(default="", description="Brief summary of the business")

class ConstraintInfo(BaseModel):
    budget: str = Field(default="", description="Budget constraints for the analysis (e.g., 10,00,000 INR)")
    timeline: str = Field(default="", description="Timeline for the analysis (e.g., 3 months)")
    team_size: str = Field(default="", description="Size of the team available for the analysis")
    other_constraints: str = Field(default="", description="Any other limitations or constraints to consider")

class ProjectBrief(BaseModel):
    # Business Context
    business_context: str = Field(..., description="Context of the project")

    # Business Metadata
    business_metadata: BusinessMetadata = Field(..., description="Metadata about the business")
    stakeholders: list[str] = Field(default_factory=list, description="List of stakeholders involved in the project")

    # Goals and Challenges
    user_goals: list[str] = Field(default_factory=list, description="Goals of the users")
    pain_points: list[str] = Field(default_factory=list, description="Pain points to address in the analysis")
    challenges: list[str] = Field(default_factory=list, description="Challenges to overcome in the analysis")

    # Dataset Information
    datasets: list[str] = Field(default_factory=list, description="File paths or URLs of datasets to be used in the analysis")

    # Constraints
    constraints: ConstraintInfo = Field(default_factory=ConstraintInfo, description="Constraints to consider for the analysis")
    tech_stack_preferences: str = Field(default="", description="Preferred technology stack for the analysis")
    format_preferences: str = Field(default="", description="Preferred formats for deliverables (e.g., PDF, Markdown)")
    audience_type: str = Field(default="", description="Type of audience for the deliverables (e.g., Executive, Technical Team, Product Team)")

    additional_context: str = Field(default="", description="Any additional context or information relevant to the project")

class BusinessUnderstanding(BaseModel):
    """
    Summarized understanding and outcomes from the initial exploration phase.
    """

    scope: str = Field(..., description="Defined scope of the business analysis project")
    risks: list[str] = Field(..., description="Identified risks associated with the project")
    key_insights: list[str] = Field(..., description="Key insights derived from the initial exploration")

class RequirementsDefinition(BaseModel):
    """
    Documented requirements and specifications for the business analysis project.
    """

    functional_requirements: list[str] = Field(..., description="Functional requirements for the project")
    non_functional_requirements: list[str] = Field(..., description="Non-functional requirements for the project")
    priorities: list[str] = Field(..., description="Prioritized list of requirements for the project")

class DataInsights(BaseModel):
    """
    Key findings and insights derived from the data analysis phase.
    """

    insights: list[str] = Field(..., description="Key insights derived from the data analysis")
    recommendations: list[str] = Field(..., description="Recommendations based on the data insights")

class GapAnalysis(BaseModel):
    """
    Documented gaps and resolutions identified during the analysis phase.
    """

    identified_gaps: list[str] = Field(..., description="Identified gaps in the analysis or data")
    root_causes: list[str] = Field(..., description="Root causes of the identified gaps")

class ActionableRecommendations(BaseModel):
    """
    Practical recommendations and action plans based on the analysis.
    """

    recommendations: list[str] = Field(..., description="Actionable recommendations for the business")
    implementation_plan: str = Field(..., description="Detailed plan for implementing the recommendations")
    expected_outcomes: list[str] = Field(..., description="Expected outcomes from implementing the recommendations")

class BusinessAnalysis(BaseModel):
    analysis: str

class RawBusinessDescription(BaseModel):
    description: str

class BusinessContextState(BaseModel):
    project_brief: ProjectBrief = None
    business_understanding: BusinessUnderstanding = None
    requirements_definition: RequirementsDefinition = None
    data_insights: DataInsights = None
    gap_analysis: GapAnalysis = None
    actionable_recommendations: ActionableRecommendations = None

class DataDependency(BaseModel):
    dataframes: Any
    dataset_metadata: str

business_context_agent = ThinkingAgent(
    config=HFModelSettings(),
    output_type=ProjectBrief,
)

@business_context_agent.system_prompt
def business_context_system_prompt(self) -> str:
    return (
        "You are an expert business data extractor. "
        "Based on the given project brief, extract and structure the business context, metadata, goals, challenges, dataset information, and constraints into a ProjectBrief format."
        "If there is file path information, retain it as is."
    )

business_understanding_agent = ThinkingAgent(
    config=HFModelSettings(),
    output_type=BusinessUnderstanding,
)

@business_understanding_agent.system_prompt
def business_understanding_system_prompt(self) -> str:
    return (
        "You are an expert business analyst. "
        "Based on the given project brief, provide a summarized understanding of the business analysis project, including scope, risks, and key insights."
    )

requirements_definition_agent = ThinkingAgent(
    config=HFModelSettings(),
    output_type=RequirementsDefinition,
)

@requirements_definition_agent.system_prompt
def requirements_definition_system_prompt(self) -> str:
    return (
        "You are an expert business analyst. "
        "Based on the given project brief, document the functional and non-functional requirements, along with their priorities for the business analysis project."
    )

analytical_query_agent = ThinkingAgent(
    config=HFModelSettings(),
    output_type=list[str]
)

@analytical_query_agent.system_prompt
def analytical_query_system_prompt(self) -> str:
    return (
        """
        You are the **Business Analyst Agent (BA)** — a strategic reasoning agent that collaborates with a **Data Analyst Agent (DA)** to explore any dataset and uncover actionable business insights.

        ---

        Given:
        - Dataset information (metadata, columns, datatypes, samples, statistics)
        - Optional business problem or objective

        Generate a **comprehensive list of analytical queries** that:
        1. Reveal insights relevant to the business context.
        2. Can be executed by the Data Analyst Agent for computation.
        3. Cover various analytical techniques (summary stats, correlations, hypothesis tests, outlier detection, etc.)
        4. Are specific, clear, and actionable.
        5. Avoid generating queries that produce excessive numerical data.
        6. Exclude any queries related to visualization or plotting.
        7. Generate maximum possible analytical queries based on the dataset and business context.
        8. Focus on queries that provide business value and actionable insights.
        9. Ensure queries are feasible given the dataset characteristics.
        10. Prioritize queries that align with user goals and pain points.
        11. Avoid redundant or overlapping queries.
        12. Ensure queries can be answered using statistical or analytical methods.
        13. Don't generate irrelevant or out-of-scope queries.
        14. Ensure queries are unbiased and objective.

        ---

        example queries:

        - "What is the correlation between customer age and purchase frequency?"
        - "Are there significant differences in average order value across different customer segments?"
        - "What are the top 5 factors contributing to customer churn?"
        - "Is there a significant association between marketing channel and conversion rate?"
        - "What is the distribution of customer lifetime value?"
        - "Are there any outliers in the sales data that could impact revenue analysis?"
        - "What are the key predictors of customer satisfaction scores?"
        - "How does seasonality affect sales performance over the past year?"
        - "What is the average time to resolution for customer support tickets?"
        - "Is there a significant difference in purchase behavior before and after a marketing campaign?"

        """
    )

coding_agent = CodingAgent(
    config=HFModelSettings(),
    output_type=str,
)

@coding_agent.system_prompt
def coding_system_prompt() -> str:
    return(
        """
        Capabilities:

        You can interpret and execute any analytical or statistical query written in plain English by automatically identifying:

        Intended operation: (e.g., aggregation, filtering, sorting, correlation, hypothesis testing)

        Relevant dataset(s): from the dataframes dictionary

        Columns and metrics involved: (e.g., sales, category, profit)

        Appropriate analytical or statistical method: for the question type

        Natural Language Query Understanding:

        You can understand and translate free-form analytical questions (e.g., “What is the most sold category?”) into accurate pandas or statistical operations.

        Data Exploration and Summary:

        You can:

        Generate descriptive statistics (mean, median, mode, std, percentiles)

        Detect missing values, duplicates, and unique counts

        Explore categorical-numerical relationships

        Statistical Analysis:

        You can perform:

        Parametric tests: t-test, ANOVA

        Non-parametric tests: chi-square and others

        Correlations: Pearson, Spearman, Kendall

        Regression: linear, logistic, and polynomial

        Compute effect sizes, p-values, and confidence intervals

        Distribution and Normality Analysis:

        You can:

        Analyze data distributions

        Perform normality tests (Shapiro-Wilk, Anderson-Darling, KS test)

        Detect skewness, kurtosis, and outliers

        Apply transformations (log, Box-Cox, Yeo-Johnson)

        Data Transformation & Feature Engineering:

        You can:

        Encode categorical variables (one-hot, label, ordinal)

        Normalize or standardize numeric features

        Handle missing values, outliers, and imbalance

        Create new derived or engineered features

        Time Series & Trend Analysis:

        You can:

        Resample data (daily, weekly, monthly)

        Identify trends, seasonality, and anomalies

        Perform stationarity tests (ADF, KPSS)

        Build forecasting models (ARIMA, SARIMA, ETS)

        Allowed Libraries:

        You may only use the following libraries for analysis:

        import pandas as pd
        import numpy as np
        from scipy import stats
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        Instructions:

        Never provide misinformation.

        Carefully read dataset details.

        Interpret the user's query precisely.

        Identify datasets, fields, and correct analytical methods.

        Write optimized Python code that works directly on pandas DataFrames.

        The final output must be stored in a variable named result.

        No plotting or visualization code should be included.

        Output Format:

        Always wrap your code inside <code>...</code> tags.

        Do not include any syntax like python or javascript.

        Output should contain only the query result, no explanations.

        Code Requirements:

        All tabular data must be returned as lists or dictionaries.

        Convert all NumPy(np.float64, np.int64, np.object_), pandas(pd.DataFrame, pd.Series), or statsmodels objects into native Python types (int, float, str, bool, list, dict).

        Round numeric values to 3 decimal places.

        Examples:

        Example - Correlation Analysis
        <code>
        correlation = dataframes['sales'][['price', 'profit']].corr()
        result = correlation.to_dict()
        </code>

        Example - T-Test
        <code>
        group1 = dataframes['customer'][dataframes['customer']['segment'] == 'A']['spend']
        group2 = dataframes['customer'][dataframes['customer']['segment'] == 'B']['spend']
        t_stat, p_value = stats.ttest_ind(group1, group2)
        result = {'t_statistic': round(t_stat, 3), 'p_value': round(p_value, 3)}
        </code>

        Example - ANOVA
        <code>
        model = ols('spend ~ C(segment)', data=dataframes['customer']).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        result = anova_table.round(3).to_dict()
        </code>
        """)

gap_analysis_agent = ThinkingAgent(
    config=HFModelSettings(),
    output_type=GapAnalysis,
)

data_insights_agent = ThinkingAgent(
    config=HFModelSettings(),
    output_type=DataInsights,
)

@data_insights_agent.system_prompt
def data_insights_system_prompt() -> str:
    return (
        "You are an expert business analyst. "
        "Based on the given project brief and data analysis results, summarize the key findings, insights, and any data quality issues identified."
    )

@gap_analysis_agent.system_prompt
def gap_analysis_system_prompt() -> str:
    return (
        "You are an expert business analyst. "
        "Based on the given project brief and data insights, document the identified gaps in the analysis or data, along with their root causes."
    )

actionable_recommendations_agent = ThinkingAgent(
    config=HFModelSettings(),
    output_type=ActionableRecommendations,
)

@actionable_recommendations_agent.system_prompt
def actionable_recommendations_system_prompt() -> str:
    return (
        "You are an expert business analyst. "
        "Based on the given project brief and data insights, provide practical recommendations and a detailed implementation plan, along with the expected outcomes."
    )

def reduce_to_str(current: str, new: str) -> str:
    return current + new

async def run_business_analysis_pipeline(description: str):
    graph = GraphBuilder(
        state_type=BusinessContextState,
        input_type=RawBusinessDescription,
        deps_type=DataDependency,
        output_type=list[Any],
    )

    collect_all_outputs = graph.join(reducer=reduce_list_extend, initial_factory=list, node_id="collect_all_outputs",)
    collect_query_outputs = graph.join(reducer=reduce_to_str, initial_factory=str, node_id="collect_query_outputs",)

    @graph.step
    async def extract_business_context(
        ctx: StepContext[BusinessContextState, DataDependency, RawBusinessDescription]
    ) -> ProjectBrief:
        result = await business_context_agent.run(ctx.inputs.description)
        return result.output

    @graph.step
    async def analyze_business_understanding(
        ctx: StepContext[BusinessContextState, DataDependency, ProjectBrief]
    ) -> tuple[ProjectBrief, BusinessUnderstanding]:
        
        project_brief = ctx.inputs

        user_prompt = f"""
        Here is the project brief:

        BUSINESS CONTEXT:
        {project_brief.business_context}

        GOALS AND CHALLENGES:
        User Goals: {', '.join(project_brief.user_goals)}
        Pain Points: {', '.join(project_brief.pain_points)}
        Challenges: {', '.join(project_brief.challenges)}

        Please analyze the business context and provide your insights.
        """
        result = await business_understanding_agent.run(user_prompt)
        return project_brief, result.output
        
    @graph.step
    async def define_requirements(
        ctx: StepContext[BusinessContextState, DataDependency, tuple[ProjectBrief, BusinessUnderstanding]]
    ) -> list[BaseModel]:
        project_brief, business_understanding = ctx.inputs
        user_prompt = f"""
        Here is the project brief:
        {project_brief.model_dump()}

        Here is the business understanding:
        {business_understanding}

        Based on this information, define the key requirements for the analysis.
        """

        result = await requirements_definition_agent.run(user_prompt)
        return [project_brief, business_understanding, result.output]
    
    @graph.step
    async def generate_analytical_queries(
        ctx: StepContext[BusinessContextState, DataDependency, tuple[ProjectBrief, BusinessUnderstanding, RequirementsDefinition]]
    ) -> list[str]:
        project_brief, business_understanding, requirements_definition = ctx.inputs

        df_paths = project_brief.datasets

        metadata_list = []
        dataframes = {}

        for path in df_paths:

            print(f"Loading dataset from: {path}")
            name = path.split("/")[-1].split(".")[0]
            df = pd.read_csv(path)
            metadata = f"""
            name : {name}\n
            num_records : {df.shape[0]}\n
            num_columns : {df.shape[1]}\n
            columns : {', '.join(df.columns.tolist())}\n
            column_data_types : {df.dtypes.apply(lambda x: x.name).to_dict()}\n
            categorial_columns : {df.select_dtypes(include=['object', 'category']).columns.tolist()}\n
            numerical_columns : {df.select_dtypes(include=[np.number]).columns.tolist()}\n
            datetime_columns : {df.select_dtypes(include=['datetime64']).columns.tolist()}\n
            sample_data : {df.head(3).to_dict(orient='records')}\n
            statistics : {df.describe(include='all').to_dict()}\n
            missing_values : {df.isnull().sum().to_dict()}\n
            """

            metadata_list.append(metadata)
            dataframes[name] = df

        dataset_info = "\n".join(metadata_list)

        user_prompt = f"""
        Here is the project brief:
        {project_brief.model_dump()}
        Business Understanding:
        {business_understanding.model_dump()}
        Requirements Definition:
        {requirements_definition.model_dump()}

        Here is the dataset information:
        {dataset_info}

        Generate a comprehensive list of analytical queries to explore the data and derive insights relevant to the business context.
        """

        result = await analytical_query_agent.run(user_prompt)

        ctx.deps.dataframes = dataframes
        ctx.deps.dataset_metadata = dataset_info

        queries = result.output

        return queries
    
    @graph.step
    async def analyze_data(
        ctx: StepContext[BusinessContextState, DataDependency, str]
    ) -> str:
        query = ctx.inputs
        dataframes = ctx.deps.dataframes
        dataset_metadata = ctx.deps.dataset_metadata

        error = None
        code = ""
        attempt = 0
        max_attempts = 5

        while True:

            if error is None:

                user_prompt = f"""
                Analyze the provided datasets according to the following user query: {query}

                Dataset Details

                Dataset Information:
                {dataset_metadata}

                Available Variables

                dataframes: A dictionary containing multiple pandas DataFrames, where each key represents a dataset name and each value is a pandas DataFrame.

                dataframes = {{name: pd.DataFrame}}
                """
            else:
                user_prompt = f"""

                code generated in previous attempt:
                {code}

                The following error was encountered during data processing: {error}

                Please adjust the analysis accordingly for the user query: {query}

                Dataset Details

                Dataset Information:
                {dataset_metadata}

                Available Variables

                dataframes: A dictionary containing multiple pandas DataFrames, where each key represents a dataset name and each value is a pandas DataFrame.

                dataframes = {{name: pd.DataFrame}}

                """

            result = await coding_agent.run(user_prompt)

            # Extract code from the response
            code_match = re.search(r"<code>(.*?)</code>", result.output, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                code = result.output.strip()

            # Execute the code safely
            local_vars = {"dataframes": dataframes, "pd": pd, "np": np, "stats": stats, "sm": sm, "ols": ols}

            try:
                exec(code, {}, local_vars)
                if "result" in local_vars:
                    result_data = local_vars["result"]
                else:
                    result_data = None
                break

            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    print(f"Max attempts reached. Last error: {str(e)}")
                    break
                print(f"Attempt {attempt} failed: {str(e)}")
                error = str(e)
                result_data = None

        result = "\n".join([query, str(result_data)]) + "\n\n"

        return result
    
    @graph.step
    async def analyzed_data_to_obj(ctx: StepContext[BusinessContextState, DataDependency, str]) -> list[BaseModel]:
        analysis_str = ctx.inputs
        return [BusinessAnalysis(analysis=analysis_str)]
    

    graph.add(
        graph.edge_from(graph.start_node).to(extract_business_context),
    )
    graph.add(
        graph.edge_from(extract_business_context).to(analyze_business_understanding),
    )
    graph.add(
        graph.edge_from(analyze_business_understanding).to(define_requirements),
    )
    graph.add(
        graph.edge_from(define_requirements).to(generate_analytical_queries, collect_all_outputs),
    )
    graph.add(
        graph.edge_from(generate_analytical_queries).map().to(analyze_data),
    )
    graph.add(
        graph.edge_from(analyze_data).to(collect_query_outputs),
    )
    graph.add(
        graph.edge_from(collect_query_outputs).to(analyzed_data_to_obj),
    )
    graph.add(
        graph.edge_from(analyzed_data_to_obj).to(collect_all_outputs),
    )
    graph.add(
        graph.edge_from(collect_all_outputs).to(graph.end_node),
    )

    graph_instance = graph.build()

    result = await graph_instance.run(
        inputs=RawBusinessDescription(description=description),
        deps=DataDependency(dataframes=None, dataset_metadata=""),
        state=BusinessContextState()
    )

    return result

if __name__ == "__main__":
    sample_description = """
    Project Context:
    DataNova Analytics, a mid-sized SaaS firm, has been facing an increasing churn rate among its subscription customers over the past year. The management suspects that churn is related to factors such as low platform engagement, delayed support responses, and pricing dissatisfaction. The company has collected detailed CRM and usage logs over the past 24 months, and now wants to analyze this data to identify churn predictors, quantify their impact, and suggest retention strategies.

    Business Metadata:

    Business Name: DataNova Analytics

    Industry: SaaS / Data Intelligence

    Company Size: Medium

    Summary: A SaaS company offering analytics dashboards and automation tools for B2B clients.

    Stakeholders:

    Head of Customer Success

    Product Manager

    Data Science Lead

    User Goals:

    Identify the key factors influencing customer churn.

    Develop a predictive model to flag high-risk customers early.

    Recommend actionable retention strategies for at-risk segments.

    Pain Points:

    Lack of visibility into customer health metrics.

    Manual reporting and delayed insights.

    No unified view across CRM, support, and usage data.

    Current Tools and Processes:
    Currently, churn analysis is done manually in Excel using exported CRM data. The process lacks automation and predictive insights.

    Datasets:

    ./customer_churn.csv (customer demographics, tenure, status)

    ./usage_logs.csv (detailed platform usage metrics)


    Metrics / KPIs:

    Monthly churn rate

    Average customer lifetime value (CLV)

    Net promoter score (NPS)

    Customer retention rate

    Constraints:

    Budget: ₹8,00,000

    Timeline: 10 weeks

    Team Size: 4 members (1 Data Engineer, 2 Analysts, 1 ML Engineer)

    Other Constraints: Must integrate with existing PostgreSQL data warehouse and export reports to Power BI.

    Tech Stack Preferences:
    Python, Pandas, SQLAlchemy, Scikit-learn, Matplotlib

    Format Preferences:
    Executive Summary PDF + Technical Markdown Report

    Audience Type:
    Mixed (Executives and Technical Team)

    Additional Context:
    The analysis should focus on actionable insights that can be implemented quickly to reduce churn within the next quarter.
        """

    import asyncio

    result = asyncio.run(run_business_analysis_pipeline(sample_description))
    print(result)


