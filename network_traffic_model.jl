using DataFrames
using CSV
using Dates
using Statistics
using Random
using MLJ
using LinearAlgebra

# Load MLJ model
RandomForestClassifier = @load RandomForestClassifier pkg="MLJDecisionTreeInterface"

"""
Structure to handle IP address features
"""
struct IPAddress
    octets::Vector{Int}
    class::Char
    is_private::Bool
    
    function IPAddress(ip_str::String)
        octets = parse.(Int, split(ip_str, "."))
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
    
    function NetworkTrafficModel()
        new(
            Standardizer(),
            RandomForestClassifier(
                n_trees=200,
                max_depth=15,
                min_samples_leaf=2,
                rng=42
            ),
            String[],
            String[],
            String[]
        )
    end
end

"""
Convert timestamp to numeric features
"""
function process_timestamp(ts::String)
    time_components = map(x -> parse(Int, x), split(ts, ":"))
    return [
        time_components[1] / 24.0,  # Hour normalized
        time_components[2] / 60.0,  # Minute normalized
        time_components[3] / 60.0   # Second normalized
    ]
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
function calculate_protocol_features(protocol::String, packet_size::Number, request_rate::Number)
    # Protocol characteristics
    header_size = if protocol == "TCP"
        20
    elseif protocol == "UDP"
        8
    else # ICMP
        8
    end
    
    # Calculate protocol-specific metrics
    payload_size = max(0, packet_size - header_size)
    overhead_ratio = header_size / packet_size
    payload_efficiency = payload_size / packet_size
    
    # Protocol risk factors
    risk_score = if protocol == "TCP"
        (packet_size < 100 && request_rate > 200) ? 1.0 : 0.5
    elseif protocol == "UDP"
        (packet_size > 250 && request_rate > 150) ? 1.0 : 0.5
    else # ICMP
        request_rate > 100 ? 1.0 : 0.5
    end
    
    return [overhead_ratio, payload_efficiency, risk_score]
end

"""
Preprocess data and engineer features
"""
function preprocess_data!(model::NetworkTrafficModel, df::DataFrame; training=true)
    if training
        model.attack_types = unique(df.Attack_Type)
        model.protocols = unique(df.Protocol)
    end
    
    n_samples = nrow(df)
    features = Vector{Float64}[]
    
    for i in 1:n_samples
        row_features = Float64[]
        
        # Process timestamp
        append!(row_features, process_timestamp(df[i, :Timestamp]))
        
        # Process IPs
        source_ip = IPAddress(df[i, :Source_IP])
        dest_ip = IPAddress(df[i, :Destination_IP])
        
        append!(row_features, extract_ip_features(source_ip))
        append!(row_features, extract_ip_features(dest_ip))
        append!(row_features, calculate_topology_features(source_ip, dest_ip))
        
        # Process protocol features
        append!(row_features, calculate_protocol_features(
            df[i, :Protocol],
            df[i, :Packet_Size],
            df[i, :Request_Rate]
        ))
        
        # Add raw metrics
        push!(row_features, Float64(df[i, :Packet_Size]))
        push!(row_features, Float64(df[i, :Request_Rate]))
        
        push!(features, row_features)
    end
    
    # Convert to matrix
    X = reduce(hcat, features)'
    
    if training
        # Create feature names
        model.feature_names = [
            "Hour", "Minute", "Second",
            "Src_Oct1", "Src_Oct2", "Src_Oct3", "Src_Oct4",
            "Src_ClassA", "Src_ClassB", "Src_ClassC", "Src_ClassD", "Src_ClassE",
            "Src_Private",
            "Dst_Oct1", "Dst_Oct2", "Dst_Oct3", "Dst_Oct4",
            "Dst_ClassA", "Dst_ClassB", "Dst_ClassC", "Dst_ClassD", "Dst_ClassE",
            "Dst_Private",
            "Subnet_Similarity", "Same_Class", "Privacy_Pattern",
            "Protocol_Overhead", "Protocol_Efficiency", "Protocol_Risk",
            "Packet_Size", "Request_Rate"
        ]
    end
    
    # Create MLJ table from features
    X_table = MLJ.table(X, names=Symbol.(model.feature_names))
    
    # Scale features
    if training
        mach = machine(model.scaler, X_table)
        fit!(mach)
        X_scaled = transform(mach, X_table)
    else
        mach = machine(model.scaler, X_table)
        X_scaled = transform(mach, X_table)
    end
    
    return X_scaled
end

"""
Train the model with cross-validation
"""
function train!(model::NetworkTrafficModel, df::DataFrame)
    println("Preprocessing data...")
    X = preprocess_data!(model, df)
    
    # Convert attack types to categorical
    y = categorical(df.Attack_Type)
    
    println("\nPerforming cross-validation...")
    cv = CV(; nfolds=5)
    mach = machine(model.model, X, y)
    cv_scores = evaluate!(mach, resampling=cv, measure=accuracy)
    
    println("Cross-validation scores: ", cv_scores.per_fold)
    println("Mean CV score: ", cv_scores.measurement[1])
    
    println("\nTraining final model...")
    fit!(mach)
    
    return Dict(
        "cv_scores" => cv_scores.per_fold,
        "mean_cv_score" => cv_scores.measurement[1]
    )
end

"""
Make predictions with detailed analysis
"""
function predict(model::NetworkTrafficModel, df::DataFrame)
    X = preprocess_data!(model, df, training=false)
    mach = machine(model.model, X)
    fit!(mach)
    
    predictions = predict(mach, X)
    probabilities = predict_mode(mach, X)
    
    # Create results DataFrame
    results = DataFrame(
        Source_IP = df.Source_IP,
        Destination_IP = df.Destination_IP,
        Protocol = df.Protocol,
        Predicted_Attack = predictions,
        Confidence = [maximum(pdf(prob)) * 100 for prob in probabilities]
    )
    
    # Add risk assessment
    results.Risk_Level = map(1:nrow(df)) do i
        confidence = results[i, :Confidence]
        risk_score = confidence / 100
        
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

# Example usage
function main()
    println("Loading data...")
    df = CSV.read("network_traffic.csv", DataFrame)
    
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
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end