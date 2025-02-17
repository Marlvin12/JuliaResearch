using DataFrames
using CSV
using Plots
using StatsPlots
using Plots.PlotMeasures
using Statistics
using StatsBase

"""
Create comprehensive visualizations for network traffic data
"""
function visualize_network_traffic(df::DataFrame)
    # Set the default plot size and style
    default(size=(800, 600), fmt=:png, dpi=300)
    
    # 1. Attack Type Distribution
    attack_dist = countmap(df.Attack_Type)
    p1 = pie(
        collect(keys(attack_dist)),
        collect(values(attack_dist)),
        title="Attack Type Distribution",
        legend=:right,
        palette=:Set2
    )
    
    # 2. Protocol Distribution by Attack Type
    p2 = groupedbar(
        df,
        :Protocol,
        group=:Attack_Type,
        title="Protocol Distribution by Attack Type",
        ylabel="Count",
        legend=:topleft,
        palette=:Set3
    )
    
    # 3. Packet Size Distribution
    p3 = histogram(
        df.Packet_Size,
        bins=30,
        title="Packet Size Distribution",
        xlabel="Packet Size",
        ylabel="Frequency",
        legend=false,
        fillalpha=0.7,
        color=:blue
    )
    
    # 4. Request Rate by Attack Type
    p4 = boxplot(
        df.Attack_Type,
        df.Request_Rate,
        title="Request Rate by Attack Type",
        xlabel="Attack Type",
        ylabel="Request Rate",
        legend=false,
        outliers=true,
        marker=(0.5, :blue, stroke(1)),
        fillalpha=0.75
    )
    
    # 5. Scatter plot: Packet Size vs Request Rate
    p5 = scatter(
        df.Packet_Size,
        df.Request_Rate,
        group=df.Attack_Type,
        title="Packet Size vs Request Rate",
        xlabel="Packet Size",
        ylabel="Request Rate",
        legend=:topleft,
        alpha=0.6,
        palette=:Set1
    )
    
    # 6. Time Series Analysis
    # Convert timestamp to hour for temporal analysis
    hours = map(x -> parse(Float64, split(x, ":")[1]), df.Timestamp)
    p6 = plot(
        hours,
        df.Request_Rate,
        group=df.Attack_Type,
        title="Request Rate Over Time",
        xlabel="Hour",
        ylabel="Request Rate",
        legend=:topleft,
        alpha=0.6,
        marker=(:circle, 4)
    )
    
    # Combine all plots
    layout = @layout [
        grid(2,2)
        a{0.5h}
        b{0.5h}
    ]
    
    final_plot = plot(
        p1, p2, p3, p4, p5, p6,
        layout=layout,
        size=(1200, 1500),
        margin=10px
    )
    
    # Save the plot
    savefig(final_plot, "network_traffic_analysis.png")
    
    # Additional Statistical Analysis
    println("\nStatistical Summary:")
    println("--------------------")
    
    # Summary by Attack Type
    println("\nAttack Type Distribution:")
    for (attack_type, count) in attack_dist
        println("$attack_type: $count ($(round(count/nrow(df)*100, digits=2))%)")
    end
    
    # Protocol Statistics
    println("\nProtocol Usage:")
    protocol_stats = combine(groupby(df, :Protocol), nrow)
    for row in eachrow(protocol_stats)
        println("$(row.Protocol): $(row.nrow) ($(round(row.nrow/nrow(df)*100, digits=2))%)")
    end
    
    # Numeric Feature Statistics
    println("\nNumeric Feature Statistics:")
    println("\nPacket Size:")
    println("Mean: $(round(mean(df.Packet_Size), digits=2))")
    println("Median: $(round(median(df.Packet_Size), digits=2))")
    println("Std Dev: $(round(std(df.Packet_Size), digits=2))")
    
    println("\nRequest Rate:")
    println("Mean: $(round(mean(df.Request_Rate), digits=2))")
    println("Median: $(round(median(df.Request_Rate), digits=2))")
    println("Std Dev: $(round(std(df.Request_Rate), digits=2))")
    
    return final_plot
end

"""
Create a correlation matrix visualization
"""
function visualize_correlations(df::DataFrame)
    # Create numeric columns for categorical variables
    df_numeric = copy(df)
    df_numeric.Attack_Type = categorical(df_numeric.Attack_Type)
    df_numeric.Attack_Type = levelcode.(df_numeric.Attack_Type)
    df_numeric.Protocol = categorical(df_numeric.Protocol)
    df_numeric.Protocol = levelcode.(df_numeric.Protocol)
    
    # Select numeric columns
    numeric_cols = [:Protocol, :Packet_Size, :Request_Rate, :Attack_Type]
    correlation_matrix = cor(Matrix(df_numeric[:, numeric_cols]))
    
    # Create correlation heatmap
    heatmap(
        correlation_matrix,
        xticks=(1:length(numeric_cols), numeric_cols),
        yticks=(1:length(numeric_cols), numeric_cols),
        title="Feature Correlation Matrix",
        xrotation=45,
        size=(800, 600),
        color=:RdBu,
        clims=(-1, 1),
        annotation=round.(correlation_matrix, digits=2)
    )
end

"""
Analyze temporal patterns
"""
function visualize_temporal_patterns(df::DataFrame)
    # Extract hour from timestamp
    hours = map(x -> parse(Float64, split(x, ":")[1]), df.Timestamp)
    
    # Create subplots for temporal analysis
    p1 = histogram(
        hours,
        bins=24,
        title="Traffic Distribution by Hour",
        xlabel="Hour",
        ylabel="Frequency",
        legend=false,
        fillalpha=0.7,
        color=:purple
    )
    
    p2 = violin(
        string.(Int.(hours)),
        df.Request_Rate,
        title="Request Rate Distribution by Hour",
        xlabel="Hour",
        ylabel="Request Rate",
        legend=false,
        fillalpha=0.5,
        color=:orange
    )
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800, 1000))
end

# Main execution
function main()
    # Read the dataset
    println("Loading data...")
    df = CSV.read("network_traffic.csv", DataFrame)
    
    println("Creating visualizations...")
    
    # Create main visualizations
    main_plot = visualize_network_traffic(df)
    
    # Create correlation matrix
    correlation_plot = visualize_correlations(df)
    savefig(correlation_plot, "correlation_matrix.png")
    
    # Create temporal analysis
    temporal_plot = visualize_temporal_patterns(df)
    savefig(temporal_plot, "temporal_patterns.png")
    
    println("Visualizations completed!")
    println("Output files: network_traffic_analysis.png, correlation_matrix.png, temporal_patterns.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end