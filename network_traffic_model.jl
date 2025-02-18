using DataFrames
using CSV
using Dates
using Statistics
using Random
using MLJ
using LinearAlgebra
using Tables
using StatsBase  

# Import specific MLJ functions to resolve ambiguity
import MLJ: transform, predict, machine, fit!

# Load XGBoost classifier
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost


"""
Structure to handle IP address features
"""
struct IPAddress
    octets::Vector{Int}
    class::Char
    is_private::Bool
    
    function IPAddress(ip_str::AbstractString)
        try
            octets = parse.(Int, split(String(ip_str), "."))
            first_octet = octets[1]
            
            # Determine IP class
            class = if first_octet < 128
                'A'
            elseif first_octet < 192
                'B'
            elseif first_octet < 224
                'C'
            elseif first_octet < 240
                'D'
            else
                'E'
            end
            
            # Check if private IP
            is_private = (
                (octets[1] == 10) ||
                (octets[1] == 172 && 16 <= octets[2] <= 31) ||
                (octets[1] == 192 && octets[2] == 168)
            )
            
            new(octets, class, is_private)
        catch e
            @warn "Error parsing IP address: $ip_str"
            new([0,0,0,0], 'A', false)  # Default values
        end
    end
end

"""
Main model structure with preprocessing components
"""
mutable struct NetworkTrafficModel
    scaler
    model
    feature_names::Vector{String}
    attack_types::Vector{String}
    protocols::Vector{String}
    machine
    function NetworkTrafficModel()
        new(
            Standardizer(; features=Symbol[]),
            XGBoostClassifier(
                num_round=3000,               # More iterations
                max_depth=8,                  # Deeper trees
                eta=0.02,                     # Lower learning rate
                gamma=0.1,                    # Minimum loss reduction
                min_child_weight=1.0,         # Allow more splits
                subsample=0.8,                # Subsample ratio
                colsample_bytree=0.8,         # Feature sampling
                lambda=0.5,                   # L2 regularization
                alpha=0.1,                    # L1 regularization
                tree_method="auto",
                scale_pos_weight=1.0,         # For imbalanced classes
                objective="multi:softprob"
            ),
            String[],
            String[],
            String[],
            nothing
        )
    end
end


"""
Convert timestamp to numeric features
"""
function process_timestamp(ts::AbstractString)
    try
        time_components = map(x -> parse(Int, x), split(String(ts), ":"))
        return [
            time_components[1] / 24.0,  # Hour normalized
            time_components[2] / 60.0,  # Minute normalized
            time_components[3] / 60.0   # Second normalized
        ]
    catch e
        @warn "Error processing timestamp: $ts"
        return [0.0, 0.0, 0.0]  # Default values
    end
end

"""
Validate and preprocess input data
"""
function validate_data(df::DataFrame)
    df_clean = copy(df)
    
    # Convert columns to appropriate types
    df_clean.Timestamp = String.(df_clean.Timestamp)
    df_clean.Source_IP = String.(df_clean.Source_IP)
    df_clean.Destination_IP = String.(df_clean.Destination_IP)
    df_clean.Protocol = String.(df_clean.Protocol)
    df_clean.Attack_Type = String.(df_clean.Attack_Type)
    df_clean.Packet_Size = Float64.(df_clean.Packet_Size)
    df_clean.Request_Rate = Float64.(df_clean.Request_Rate)
    
    return df_clean
end

"""
Extract features from IP address
"""
function extract_ip_features(ip::IPAddress)
    features = Float64[]
    
    # Normalized octets
    append!(features, ip.octets ./ 255.0)
    
    # One-hot encoded class
    class_encoding = zeros(Float64, 5)
    class_idx = findfirst(==(ip.class), ['A', 'B', 'C', 'D', 'E'])
    class_encoding[class_idx] = 1.0
    append!(features, class_encoding)
    
    # Private IP flag
    push!(features, Float64(ip.is_private))
    
    return features
end

"""
Calculate network topology features
"""
function calculate_topology_features(source_ip::IPAddress, dest_ip::IPAddress)
    # Subnet similarity (matching octets)
    subnet_similarity = sum(source_ip.octets .== dest_ip.octets) / 4.0
    
    # Class relationship
    same_class = Float64(source_ip.class == dest_ip.class)
    
    # Privacy relationship
    privacy_pattern = if !source_ip.is_private && dest_ip.is_private
        1.0  # Public to Private (potentially suspicious)
    elseif source_ip.is_private && !dest_ip.is_private
        0.5  # Private to Public (less suspicious)
    else
        0.0  # Same privacy level
    end
    
    return [subnet_similarity, same_class, privacy_pattern]
end

"""
Calculate protocol-specific features
"""
function calculate_protocol_features(protocol::AbstractString, packet_size::Number, request_rate::Number)
    # Protocol characteristics
    header_size = if String(protocol) == "TCP"
        20
    elseif String(protocol) == "UDP"
        8
    else # ICMP
        8
    end
    
    # Calculate protocol-specific metrics
    payload_size = max(0, packet_size - header_size)
    overhead_ratio = header_size / packet_size
    payload_efficiency = payload_size / packet_size
    
    # Protocol risk factors - adjusted based on your data patterns
    risk_score = if String(protocol) == "TCP"
        if packet_size < 100 && request_rate > 200
            1.0  # DoS signature
        elseif packet_size > 300 && request_rate > 300
            0.8  # DDoS signature
        elseif packet_size > 200 && request_rate < 100
            0.3  # Normal traffic
        else
            0.5  # Probe or uncertain
        end
    elseif String(protocol) == "UDP"
        if packet_size > 400 || request_rate > 400
            0.9  # DDoS signature
        elseif packet_size < 50 && request_rate > 300
            0.8  # DoS signature
        else
            0.4  # Normal or Probe
        end
    else # ICMP
        if request_rate > 300
            0.9  # Attack signature
        elseif packet_size > 300
            0.7  # Suspicious
        else
            0.3  # Normal
        end
    end
    
    return [overhead_ratio, payload_efficiency, risk_score]
end

"""
Custom standardization with handling for low variance
"""
function custom_standardize(X::AbstractMatrix)
    means = mean(X, dims=1)
    stds = std(X, dims=1)
    
    # Replace very small standard deviations with 1.0
    stds[stds .< 1e-10] .= 1.0
    
    # Standardize
    X_scaled = (X .- means) ./ stds
    return X_scaled
end

"""
Create additional features for the model
"""
function create_additional_features(df::DataFrame)
    # Create copy to avoid modifying original
    df_new = copy(df)
    
    # Request rate percentiles for normalization
    rate_p95 = percentile(df_new.Request_Rate, 95)
    
    # Add engineered features
    df_new.Normalized_Request_Rate = df_new.Request_Rate ./ rate_p95
    df_new.Bytes_Per_Second = df_new.Packet_Size .* df_new.Request_Rate
    df_new.High_Rate_Flag = Int.(df_new.Request_Rate .> median(df_new.Request_Rate))
    
    # Protocol frequencies
    protocol_freqs = countmap(df_new.Protocol)
    df_new.Protocol_Freq = [protocol_freqs[p] for p in df_new.Protocol]
    
    # Time-based features from timestamp
    df_new.Hour = parse.(Int, first.(split.(df_new.Timestamp, ":")))
    df_new.Is_Peak_Hour = Int.(df_new.Hour .>= 9 .&& df_new.Hour .<= 17)
    
    return df_new
end

function preprocess_data!(model::NetworkTrafficModel, df::DataFrame; training=true)
    # Add engineered features
    df_enriched = create_additional_features(df)
    
    if training
        model.attack_types = unique(df_enriched.Attack_Type)
        model.protocols = unique(df_enriched.Protocol)
    end
    
    # Create features
    features = map(1:nrow(df_enriched)) do i
        row = df_enriched[i, :]
        row_features = Float64[]
        
        # Basic numeric features
        append!(row_features, [
            row.Packet_Size,
            row.Request_Rate,
            row.Normalized_Request_Rate,
            row.Bytes_Per_Second,
            row.High_Rate_Flag,
            row.Protocol_Freq,
            row.Is_Peak_Hour
        ])
        
        # Process IPs
        source_ip = IPAddress(row.Source_IP)
        dest_ip = IPAddress(row.Destination_IP)
        
        # IP-based features
        append!(row_features, extract_ip_features(source_ip))
        append!(row_features, extract_ip_features(dest_ip))
        append!(row_features, calculate_topology_features(source_ip, dest_ip))
        
        # Protocol features
        append!(row_features, calculate_protocol_features(
            row.Protocol,
            row.Packet_Size,
            row.Request_Rate
        ))
        
        row_features
    end
    
    # Convert to matrix
    X = reduce(hcat, features)'
    
    if training
        # Update feature names
        model.feature_names = [
            "Packet_Size", "Request_Rate", "Normalized_Request_Rate",
            "Bytes_Per_Second", "High_Rate_Flag", "Protocol_Freq",
            "Is_Peak_Hour",
            "Src_Oct1", "Src_Oct2", "Src_Oct3", "Src_Oct4",
            "Src_ClassA", "Src_ClassB", "Src_ClassC", "Src_ClassD", "Src_ClassE",
            "Src_Private",
            "Dst_Oct1", "Dst_Oct2", "Dst_Oct3", "Dst_Oct4",
            "Dst_ClassA", "Dst_ClassB", "Dst_ClassC", "Dst_ClassD", "Dst_ClassE",
            "Dst_Private",
            "Subnet_Similarity", "Same_Class", "Privacy_Pattern",
            "Protocol_Overhead", "Protocol_Efficiency", "Protocol_Risk"
        ]
    end
    
    # Create MLJ table and scale features
    X_scaled = custom_standardize(X)
    X_table = MLJ.table(X_scaled, names=Symbol.(model.feature_names))
    
    return X_table
end


function MLJ.predict(model::NetworkTrafficModel, df::DataFrame)
    if isnothing(model.machine)
        error("Model must be trained before making predictions")
    end
    
    # Preprocess input data
    X = preprocess_data!(model, df, training=false)
    X_df = DataFrame(X)
    
    # Get predictions and probabilities
    predictions = predict_mode(model.machine, X_df)
    probs = MLJ.predict(model.machine, X_df)
    
    # Extract probabilities correctly
    confidences = map(probs) do prob
        maximum(values(prob.prob_given_ref))
    end
    
    # Create results DataFrame
    results = DataFrame(
        Source_IP = df.Source_IP,
        Destination_IP = df.Destination_IP,
        Protocol = df.Protocol,
        Predicted_Attack = predictions,
        Confidence = confidences .* 100
    )
    
    # Add risk assessment
    results.Risk_Level = map(1:nrow(df)) do i
        row = df[i, :]
        
        # Get various risk factors
        protocol_risk = calculate_protocol_features(
            row.Protocol,
            row.Packet_Size,
            row.Request_Rate
        )[3]
        
        source_ip = IPAddress(row.Source_IP)
        dest_ip = IPAddress(row.Destination_IP)
        network_risk = calculate_topology_features(source_ip, dest_ip)[3]
        confidence_risk = confidences[i]
        
        risk_score = 0.4 * protocol_risk + 0.3 * network_risk + 0.3 * confidence_risk
        
        if risk_score > 0.8
            "High"
        elseif risk_score > 0.5
            "Medium"
        else
            "Low"
        end
    end
    
    return results
end

function train!(model::NetworkTrafficModel, df::DataFrame)
    try
        println("Preprocessing data...")
        X = preprocess_data!(model, df)
        X_df = DataFrame(X)
        y = categorical(df.Attack_Type)
        
        # Calculate class weights for scale_pos_weight parameter
        class_counts = countmap(y)
        total_samples = sum(values(class_counts))
        class_weights = Dict(k => total_samples/(length(class_counts) * v) for (k,v) in class_counts)
        
        # Update model parameters with class weights
        model.model.scale_pos_weight = mean(values(class_weights))
        
        println("\nTraining model...")
        mach = machine(model.model, X_df, y)
        fit!(mach, rows=1:nrow(df), verbosity=0)  # Removed weights parameter
        model.machine = mach
        
        # Evaluate performance
        y_pred = predict_mode(mach, X_df)
        println("\nTraining Performance:")
        for class in unique(y)
            idx = y .== class
            accuracy = mean(y_pred[idx] .== y[idx])
            precision = sum(y_pred[idx] .== class) / (sum(y_pred .== class) + eps())
            recall = sum(y_pred[idx] .== class) / sum(idx)
            f1 = 2 * (precision * recall) / (precision + recall + eps())
            
            println("\n$class:")
            println("  Accuracy: $(round(accuracy * 100, digits=1))%")
            println("  Precision: $(round(precision * 100, digits=1))%")
            println("  Recall: $(round(recall * 100, digits=1))%")
            println("  F1 Score: $(round(f1 * 100, digits=1))%")
            println("  Samples: $(sum(idx))")
        end
        
        return mach
    catch e
        println("\nError during training:")
        println(e)
        rethrow(e)
    end
end

function main()
    try
        println("Loading data...")
        df = CSV.read("network_traffic.csv", DataFrame)
        
        println("\nData Overview:")
        println("Number of records: ", nrow(df))
        println("Attack Types: ", unique(df.Attack_Type))
        println("Protocols: ", unique(df.Protocol))
        
        println("\nInitializing model...")
        model = NetworkTrafficModel()
        
        println("\nTraining model...")
        train_results = train!(model, df)
        
        println("\nMaking predictions...")
        predictions = predict(model, df)
        
        println("\nSample predictions:")
        first_predictions = first(predictions, 5)
        for row in eachrow(first_predictions)
            println("\nSource IP: $(row.Source_IP) → Destination IP: $(row.Destination_IP)")
            println("Protocol: $(row.Protocol)")
            println("Predicted Attack: $(row.Predicted_Attack)")
            println("Confidence: $(round(row.Confidence, digits=1))%")
            println("Risk Level: $(row.Risk_Level)")
        end
        
    catch e
        println("\nError occurred during execution:")
        println(e)
        println("\nStacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end