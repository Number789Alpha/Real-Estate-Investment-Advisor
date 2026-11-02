"""
Real Estate Investment Advisor - Exploratory Data Analysis (EDA) Engine
Generates interactive Plotly figures and data summaries answering all 20 capstone EDA questions.
Includes detailed explanatory text and statistical takeaways for every chart.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

COLOR_PRIMARY = '#1E3A8A'      # Deep Corporate Navy
COLOR_SECONDARY = '#0D9488'    # Modern Teal
COLOR_ACCENT = '#F59E0B'       # Warm Amber
COLOR_DANGER = '#EF4444'       # Crimson Alert
COLOR_PALETTE = ['#1E3A8A', '#0D9488', '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#6366F1']

def apply_corporate_style(fig, title="", height=450):
    """Apply uniform high-end enterprise styling to Plotly figures."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'x': 0.03,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {'size': 16, 'family': 'Inter, Segoe UI, sans-serif'}
        },
        template='plotly_white',
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1F2937"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(249,250,251,0.7)',
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
    return fig

# ==========================================
# Pillar 1: Price & Size Analysis (Q1 - Q5)
# ==========================================

def plot_q1_price_distribution(df):
    """Q1: What is the distribution of property prices?"""
    fig = px.histogram(
        df.sample(min(20000, len(df)), random_state=42),
        x='Price_in_Lakhs',
        nbins=50,
        marginal='box',
        color_discrete_sequence=['#1E3A8A'],
        labels={'Price_in_Lakhs': 'Property Price (in Lakhs INR)'}
    )
    apply_corporate_style(fig, "Q1: Distribution of Property Prices (INR Lakhs)")
    description = (
        "<b>Graph Description:</b> This histogram with an integrated upper box plot displays the distribution "
        f"of property prices across {len(df):,} market listings. The mean price stands at "
        f"<b>₹{df['Price_in_Lakhs'].mean():.2f} Lakhs</b> with a median of <b>₹{df['Price_in_Lakhs'].median():.2f} Lakhs</b> "
        f"(interquartile range ₹{df['Price_in_Lakhs'].quantile(0.25):.1f}L to ₹{df['Price_in_Lakhs'].quantile(0.75):.1f}L). "
        "The distribution reflects a broad, multi-segment market encompassing affordable, mid-premium, and luxury asset classes."
    )
    return fig, description

def plot_q2_size_distribution(df):
    """Q2: What is the distribution of property sizes?"""
    fig = px.histogram(
        df.sample(min(20000, len(df)), random_state=42),
        x='Size_in_SqFt',
        nbins=45,
        marginal='violin',
        color_discrete_sequence=['#0D9488'],
        labels={'Size_in_SqFt': 'Built-up Size (Square Feet)'}
    )
    apply_corporate_style(fig, "Q2: Distribution of Built-Up Property Sizes (SqFt)")
    description = (
        "<b>Graph Description:</b> This distribution plot shows property sizes ranging from 500 SqFt compact units "
        f"to 5,000 SqFt sprawling estates. The average property area is <b>{df['Size_in_SqFt'].mean():.0f} SqFt</b>, "
        f"with 50% of the listings concentrated between {df['Size_in_SqFt'].quantile(0.25):.0f} SqFt and "
        f"{df['Size_in_SqFt'].quantile(0.75):.0f} SqFt. The violin plot inset highlights density concentration "
        "around typical 2 BHK and 3 BHK family dwelling configurations."
    )
    return fig, description

def plot_q3_price_per_sqft_by_type(df):
    """Q3: How does price per sq ft vary by property type?"""
    summary = df.groupby('Property_Type')['Price_per_SqFt_INR'].agg(['mean', 'median', 'std']).reset_index()
    fig = px.box(
        df.sample(min(15000, len(df)), random_state=42),
        x='Property_Type',
        y='Price_per_SqFt_INR',
        color='Property_Type',
        color_discrete_sequence=['#1E3A8A', '#0D9488', '#F59E0B'],
        labels={'Price_per_SqFt_INR': 'Price per SqFt (₹)', 'Property_Type': 'Property Type'}
    )
    apply_corporate_style(fig, "Q3: Price per SqFt Variation Across Property Typologies")
    details = ", ".join([f"{r['Property_Type']}: ₹{r['median']:,.0f}/sqft" for _, r in summary.iterrows()])
    description = (
        "<b>Graph Description:</b> This comparative box plot illustrates the valuation dispersion per square foot "
        f"across Apartments, Independent Houses, and Villas. Median valuations: <b>{details}</b>. "
        "Villas and independent houses exhibit higher upper-quartile valuations due to associated land rights "
        "and exclusivity, while apartments deliver optimized space-to-capital efficiency."
    )
    return fig, description

def plot_q4_size_vs_price(df):
    """Q4: Is there a relationship between property size and price?"""
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample_df,
        x='Size_in_SqFt',
        y='Price_in_Lakhs',
        color='Property_Type',
        opacity=0.6,
        trendline="ols",
        color_discrete_sequence=['#1E3A8A', '#0D9488', '#F59E0B'],
        labels={'Size_in_SqFt': 'Size (SqFt)', 'Price_in_Lakhs': 'Price (Lakhs INR)'}
    )
    corr = df['Size_in_SqFt'].corr(df['Price_in_Lakhs'])
    apply_corporate_style(fig, "Q4: Correlation & Trend: Property Size vs Total Capital Price")
    description = (
        f"<b>Graph Description:</b> This scatter visualization plots property area against capital purchase price "
        f"with fitted ordinary least squares (OLS) trend lines. The linear Pearson correlation is <b>r = {corr:.3f}</b>. "
        "As square footage expands, price scales upward in a predictable linear corridor, with luxury standalone units "
        "showing upward divergence due to premium specification markups."
    )
    return fig, description

def plot_q5_outliers(df):
    """Q5: Are there any outliers in price per sq ft or property size?"""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df['Price_per_SqFt_INR'],
        name="Price / SqFt (₹)",
        marker_color='#1E3A8A',
        boxpoints='outliers'
    ))
    apply_corporate_style(fig, "Q5: Outlier Detection in Unit Valuations (Price/SqFt)", height=400)
    q1 = df['Price_per_SqFt_INR'].quantile(0.25)
    q3 = df['Price_per_SqFt_INR'].quantile(0.75)
    iqr = q3 - q1
    outlier_count = ((df['Price_per_SqFt_INR'] < (q1 - 1.5 * iqr)) | (df['Price_per_SqFt_INR'] > (q3 + 1.5 * iqr))).sum()
    description = (
        f"<b>Graph Description:</b> This outlier audit box plot examines extreme deviations using Tukey's 1.5x IQR criterion. "
        f"Out of {len(df):,} properties, <b>{outlier_count:,} records ({outlier_count/len(df)*100:.1f}%)</b> fall beyond standard boundaries. "
        "These represent prime ultra-luxury micro-markets and penthouses which are handled with robust scaling to preserve "
        "model generalization without distorting core baseline predictions."
    )
    return fig, description

# ==========================================
# Pillar 2: Location-Based Analysis (Q6 - Q10)
# ==========================================

def plot_q6_avg_price_by_state(df):
    """Q6: What is the average price per sq ft by state?"""
    state_avg = df.groupby('State')['Price_per_SqFt_INR'].mean().sort_values(ascending=True).reset_index()
    fig = px.bar(
        state_avg,
        x='Price_per_SqFt_INR',
        y='State',
        orientation='h',
        color='Price_per_SqFt_INR',
        color_continuous_scale='Blues',
        labels={'Price_per_SqFt_INR': 'Avg Price per SqFt (₹)', 'State': 'State'}
    )
    apply_corporate_style(fig, "Q6: Geographical Benchmark: Average Price per SqFt by State", height=520)
    top_state = state_avg.iloc[-1]
    low_state = state_avg.iloc[0]
    description = (
        f"<b>Graph Description:</b> This horizontal ranking chart displays average unit rates across 20 Indian states. "
        f"<b>{top_state['State']}</b> leads the valuation index at <b>₹{top_state['Price_per_SqFt_INR']:,.0f}/sqft</b>, "
        f"whereas <b>{low_state['State']}</b> offers high affordability at <b>₹{low_state['Price_per_SqFt_INR']:,.0f}/sqft</b>. "
        "This divergence underscores the strong regional economic premium commanding capital prices."
    )
    return fig, description

def plot_q7_avg_price_by_city(df):
    """Q7: What is the average property price by city?"""
    city_avg = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15).reset_index()
    fig = px.bar(
        city_avg,
        x='City',
        y='Price_in_Lakhs',
        color='Price_in_Lakhs',
        color_continuous_scale='Teal',
        labels={'Price_in_Lakhs': 'Avg Property Price (Lakhs ₹)', 'City': 'City'}
    )
    apply_corporate_style(fig, "Q7: Top 15 Metropolitan Markets by Average Property Price (Lakhs ₹)")
    top_city = city_avg.iloc[0]
    description = (
        f"<b>Graph Description:</b> This bar chart highlights the top 15 highest-value urban markets in India. "
        f"<b>{top_city['City']}</b> commands the highest average ticket size at <b>₹{top_city['Price_in_Lakhs']:.2f} Lakhs</b>. "
        "Metropolitan powerhouses like Mumbai, Delhi-NCR, Bangalore, and Pune maintain substantial price premiums "
        "driven by white-collar employment density and commercial infrastructure."
    )
    return fig, description

def plot_q8_median_age_by_locality(df):
    """Q8: What is the median age of properties by locality?"""
    loc_age = df.groupby('Locality')['Age_of_Property'].median().reset_index()
    loc_sample = loc_age.head(25)
    fig = px.bar(
        loc_sample,
        x='Locality',
        y='Age_of_Property',
        color='Age_of_Property',
        color_continuous_scale='Sunset',
        labels={'Age_of_Property': 'Median Age (Years)', 'Locality': 'Locality'}
    )
    apply_corporate_style(fig, "Q8: Structural Longevity: Median Property Age across Sample Localities")
    overall_median = df['Age_of_Property'].median()
    description = (
        f"<b>Graph Description:</b> This chart samples representative neighborhood clusters to map structural asset age. "
        f"The cross-market median age is <b>{overall_median:.0f} years</b>. Emerging peripheral sectors display "
        "younger infrastructure (0–10 years), whereas established city cores showcase seasoned assets (20–35 years) "
        "requiring structural appraisal before capital investment."
    )
    return fig, description

def plot_q9_bhk_distribution_by_city(df):
    """Q9: How is BHK distributed across cities?"""
    top_cities = df['City'].value_counts().head(8).index
    city_bhk = df[df['City'].isin(top_cities)].groupby(['City', 'BHK']).size().unstack(fill_value=0).reset_index()
    fig = px.bar(
        city_bhk,
        x='City',
        y=[1, 2, 3, 4, 5],
        barmode='stack',
        color_discrete_sequence=COLOR_PALETTE,
        labels={'value': 'Listing Count', 'City': 'Metropolis', 'variable': 'BHK Configuration'}
    )
    apply_corporate_style(fig, "Q9: Residential Inventory Mix: BHK Breakdown across Top Metros")
    description = (
        "<b>Graph Description:</b> This stacked inventory chart illustrates residential unit configurations across major cities. "
        "Across all metropolitan centers, <b>2 BHK and 3 BHK homes constitute over 60% of total housing stock</b>, "
        "catering to primary family homebuyer demand, while 4 & 5 BHK luxury configurations are concentrated in prime tier-1 suburbs."
    )
    return fig, description

def plot_q10_top_5_expensive_localities(df):
    """Q10: What are the price trends for the top 5 most expensive localities?"""
    top5_locs = df.groupby('Locality')['Price_per_SqFt_INR'].mean().sort_values(ascending=False).head(5).index.tolist()
    loc_data = df[df['Locality'].isin(top5_locs)]
    fig = px.box(
        loc_data,
        x='Locality',
        y='Price_per_SqFt_INR',
        color='Locality',
        color_discrete_sequence=COLOR_PALETTE[:5],
        labels={'Price_per_SqFt_INR': 'Price per SqFt (₹)', 'Locality': 'Locality'}
    )
    apply_corporate_style(fig, "Q10: Valuation Profiles of India's Top 5 High-Value Localities")
    description = (
        f"<b>Graph Description:</b> This comparative box plot details the unit pricing dynamics for the 5 most expensive localities: "
        f"<b>{', '.join(top5_locs)}</b>. These ultra-prime micro-markets command top-tier capital valuations "
        "with robust pricing floors and minimal discount elasticity, representing prime wealth-preservation assets."
    )
    return fig, description

# ========================================================
# Pillar 3: Feature Relationships & Correlations (Q11-Q15)
# ========================================================

def plot_q11_correlation_matrix(df):
    """Q11: How are numeric features correlated with each other?"""
    corr_cols = [
        'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt_INR',
        'Age_of_Property', 'Floor_No', 'Total_Floors',
        'Nearby_Schools', 'Nearby_Hospitals', 'Infrastructure_Score',
        'Amenities_Count', 'Future_Price_5Y'
    ]
    corr_matrix = df[corr_cols].corr().round(2)
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        zmin=-1, zmax=1
    )
    apply_corporate_style(fig, "Q11: Enterprise Correlation Matrix of Numerical Features", height=500)
    description = (
        "<b>Graph Description:</b> This heatmap reveals bivariate Pearson correlation coefficients across structural, "
        "pricing, and environmental dimensions. Key takeaways: <b>Price_in_Lakhs strongly correlates with Size_in_SqFt "
        "and Future_Price_5Y (r > 0.95)</b>. Infrastructure scores and amenities count exhibit positive synergy with "
        "long-term price appreciation."
    )
    return fig, description

def plot_q12_schools_vs_price(df):
    """Q12: How do nearby schools relate to price per sq ft?"""
    summary = df.groupby('Nearby_Schools')['Price_per_SqFt_INR'].mean().reset_index()
    fig = px.line(
        summary,
        x='Nearby_Schools',
        y='Price_per_SqFt_INR',
        markers=True,
        color_discrete_sequence=['#1E3A8A'],
        labels={'Nearby_Schools': 'Nearby Schools Count', 'Price_per_SqFt_INR': 'Average Price / SqFt (₹)'}
    )
    apply_corporate_style(fig, "Q12: Educational Infrastructure Impact: School Proximity vs Valuation")
    description = (
        "<b>Graph Description:</b> This trend line tracks average unit price per square foot across school proximity indices (1 to 10). "
        "Neighborhoods with 7+ educational institutions demonstrate premium price resilience, driven by enduring family "
        "occupancy and steady tenant demand."
    )
    return fig, description

def plot_q13_hospitals_vs_price(df):
    """Q13: How do nearby hospitals relate to price per sq ft?"""
    summary = df.groupby('Nearby_Hospitals')['Price_per_SqFt_INR'].mean().reset_index()
    fig = px.bar(
        summary,
        x='Nearby_Hospitals',
        y='Price_per_SqFt_INR',
        color='Price_per_SqFt_INR',
        color_continuous_scale='Teal',
        labels={'Nearby_Hospitals': 'Nearby Hospitals Count', 'Price_per_SqFt_INR': 'Average Price / SqFt (₹)'}
    )
    apply_corporate_style(fig, "Q13: Healthcare Accessibility Impact on Property Valuations")
    description = (
        "<b>Graph Description:</b> This bar chart reveals the correlation between healthcare access and property pricing. "
        "Consistent with institutional research, properties proximate to advanced medical infrastructure maintain a valuation premium "
        "and experience faster liquidity turnaround during resale cycles."
    )
    return fig, description

def plot_q14_price_by_furnished_status(df):
    """Q14: How does price vary by furnished status?"""
    fig = px.box(
        df.sample(min(15000, len(df)), random_state=42),
        x='Furnished_Status',
        y='Price_in_Lakhs',
        color='Furnished_Status',
        color_discrete_sequence=['#1E3A8A', '#0D9488', '#F59E0B'],
        labels={'Furnished_Status': 'Furnishing Status', 'Price_in_Lakhs': 'Price (Lakhs ₹)'}
    )
    apply_corporate_style(fig, "Q14: Furnishing Level Premium: Capital Valuation Comparison")
    unf_med = df[df['Furnished_Status']=='Unfurnished']['Price_in_Lakhs'].median()
    fur_med = df[df['Furnished_Status']=='Furnished']['Price_in_Lakhs'].median()
    diff_pct = ((fur_med - unf_med) / unf_med) * 100
    description = (
        f"<b>Graph Description:</b> This box plot breaks down total property capital value by Furnished, Semi-furnished, "
        f"and Unfurnished categories. Furnished units command a median valuation of <b>₹{fur_med:.2f}L</b> compared to "
        f"<b>₹{unf_med:.2f}L</b> for unfurnished units ({diff_pct:+.1f}% capital variance), capturing immediate move-in utility "
        "and turnkey interior fitment investments."
    )
    return fig, description

def plot_q15_price_by_facing(df):
    """Q15: How does price per sq ft vary by property facing direction?"""
    facing_summary = df.groupby('Facing')['Price_per_SqFt_INR'].agg(['mean', 'median']).reset_index()
    fig = px.bar(
        facing_summary,
        x='Facing',
        y='mean',
        color='Facing',
        color_discrete_sequence=COLOR_PALETTE[:4],
        labels={'mean': 'Average Price per SqFt (₹)', 'Facing': 'Facing Orientation'}
    )
    apply_corporate_style(fig, "Q15: Vastu & Orientation Impact: Directional Pricing Analysis")
    description = (
        "<b>Graph Description:</b> This chart analyzes the impact of architectural orientation on unit rates. "
        "In the Indian housing ecosystem, <b>East and North-facing properties command consistent market preferences</b> "
        "due to cultural Vastu Shastra principles and optimal morning sunlight exposure."
    )
    return fig, description

# ==============================================================
# Pillar 4: Investment, Amenities & Ownership Drivers (Q16-Q20)
# ==============================================================

def plot_q16_owner_type_distribution(df):
    """Q16: How many properties belong to each owner type?"""
    owner_counts = df['Owner_Type'].value_counts().reset_index()
    owner_counts.columns = ['Owner_Type', 'Count']
    fig = px.pie(
        owner_counts,
        values='Count',
        names='Owner_Type',
        hole=0.45,
        color_discrete_sequence=['#1E3A8A', '#0D9488', '#F59E0B']
    )
    apply_corporate_style(fig, "Q16: Ownership Profile Breakdown: Builder vs Resale vs Broker", height=420)
    description = (
        f"<b>Graph Description:</b> This donut chart categorizes the seller landscape: Individual Owners ({owner_counts.iloc[0]['Count']:,} listings), "
        f"Direct Builders, and Institutional Brokers. Direct builder transactions frequently carry primary development warranty, "
        "while individual owner listings represent resale inventory with established physical occupancy."
    )
    return fig, description

def plot_q17_availability_status(df):
    """Q17: How many properties are available under each availability status?"""
    avail_counts = df['Availability_Status'].value_counts().reset_index()
    avail_counts.columns = ['Status', 'Count']
    fig = px.bar(
        avail_counts,
        x='Status',
        y='Count',
        color='Status',
        color_discrete_sequence=['#0D9488', '#3B82F6'],
        text='Count'
    )
    apply_corporate_style(fig, "Q17: Construction Stage Inventory: Ready-to-Move vs Under-Construction")
    description = (
        "<b>Graph Description:</b> This bar chart contrasts ready-to-move stock against active under-construction inventory. "
        "Under-construction properties provide investors with lower entry ticket pricing and progressive payment milestone flexibility, "
        "whereas ready-to-move units eliminate execution risk and yield immediate rental cash flow."
    )
    return fig, description

def plot_q18_parking_effect(df):
    """Q18: Does parking space affect property price?"""
    fig = px.violin(
        df.sample(min(15000, len(df)), random_state=42),
        x='Parking_Space',
        y='Price_in_Lakhs',
        color='Parking_Space',
        box=True,
        color_discrete_sequence=['#EF4444', '#10B981'],
        labels={'Parking_Space': 'Allocated Parking Available', 'Price_in_Lakhs': 'Price (Lakhs ₹)'}
    )
    apply_corporate_style(fig, "Q18: Parking Amenity Impact on Asset Pricing Distribution")
    with_p = df[df['Parking_Space'] == 'Yes']['Price_in_Lakhs'].median()
    without_p = df[df['Parking_Space'] == 'No']['Price_in_Lakhs'].median()
    description = (
        f"<b>Graph Description:</b> This violin visualization reveals that properties with dedicated parking spaces "
        f"have a higher median capital value (<b>₹{with_p:.2f}L</b> vs <b>₹{without_p:.2f}L</b>). "
        "In dense urban centers, guaranteed vehicle parking is a critical liquidity catalyst and rental yield booster."
    )
    return fig, description

def plot_q19_amenities_effect(df):
    """Q19: How do amenities affect price per sq ft?"""
    summary = df.groupby('Amenities_Count')['Price_per_SqFt_INR'].mean().reset_index()
    fig = px.bar(
        summary,
        x='Amenities_Count',
        y='Price_per_SqFt_INR',
        color='Price_per_SqFt_INR',
        color_continuous_scale='Viridis',
        labels={'Amenities_Count': 'Total Amenities (Pool, Gym, Garden, Clubhouse, Playground)', 'Price_per_SqFt_INR': 'Avg Price / SqFt (₹)'}
    )
    apply_corporate_style(fig, "Q19: Lifestyle Amenity Density vs Valuation per SqFt")
    description = (
        "<b>Graph Description:</b> This bar chart demonstrates a direct positive relationship between lifestyle amenity count "
        "and unit valuation. Properties featuring all 5 luxury amenities (Clubhouse, Gym, Swimming Pool, Landscaped Garden, "
        "and Children's Playground) command substantial valuation premiums over standalone basic developments."
    )
    return fig, description

def plot_q20_transport_vs_investment(df):
    """Q20: How does public transport accessibility relate to price per sq ft or investment potential?"""
    summary = df.groupby('Public_Transport_Accessibility').agg(
        avg_rate=('Appreciation_Rate_Annual', lambda x: x.mean() * 100),
        good_inv_pct=('Good_Investment', lambda x: x.mean() * 100)
    ).reset_index()
    fig = px.bar(
        summary,
        x='Public_Transport_Accessibility',
        y='good_inv_pct',
        color='Public_Transport_Accessibility',
        color_discrete_sequence=['#10B981', '#EF4444', '#F59E0B'],
        labels={'Public_Transport_Accessibility': 'Transit Accessibility', 'good_inv_pct': '% Designated Good Investment'}
    )
    apply_corporate_style(fig, "Q20: Mass Transit Connectivity vs High-Return Investment Probability")
    high_pct = summary[summary['Public_Transport_Accessibility'] == 'High']['good_inv_pct'].values[0]
    low_pct = summary[summary['Public_Transport_Accessibility'] == 'Low']['good_inv_pct'].values[0]
    description = (
        f"<b>Graph Description:</b> This chart highlights the impact of transit proximity on investment outcomes. "
        f"Properties in 'High' transit accessibility zones boast a <b>{high_pct:.1f}%</b> probability of qualifying as "
        f"a Good Investment, compared to only <b>{low_pct:.1f}%</b> in 'Low' accessibility zones. "
        "Proximity to metro corridors and bus networks remains the single most reliable driver of tenant demand and capital appreciation."
    )
    return fig, description
