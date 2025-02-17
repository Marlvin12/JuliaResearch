using DataFrames
using CSV
using Dates
using Statistics
using Random
using ScikitLearn
using MLJ
using LinearAlgebra

@sk_import preprocessing: StandardScaler
@sk_import ensemble: RandomForestClassifier
@sk_import model_selection: cross_val_score
@sk_import metrics: confusion_matrix
@sk_import metrics: classification_report

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
    scaler::StandardScaler
    model::RandomForestClassifier
    feature_names::Vector{String}
    attack_types::Vector{String}
    protocols::Vector{String}
    
    function NetworkTrafficModel()
        new(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced"
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
    
    for i in 1:n